import torch
import torch.nn.functional as functional
import torch.nn as nn

import numpy as np
import imageio

import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os, sys

from networks import MLP
import visualizations
import utils

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore


def compare_interpolation(image_url=None, scale_factor=2):
    if image_url is None:
        image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]
    img = torch.from_numpy(img).float().permute(2, 0, 1).cuda()
    img_coarse = img[:, ::scale_factor, ::scale_factor]

    gridDimensions = np.array([512//scale_factor, 512//scale_factor])
    domain = np.array([[0., 1.], [0., 1.]])
    mgrid = utils.MeshGrid(gridDimensions, domain, flatten=False)
    x = next(iter(mgrid))
    x = x.cuda()

    model = MLP(2, 3 ,256, 4, 256, 10, nn.ReLU(), nn.Sigmoid()).cuda()

    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

    for i in tqdm(range(10000), desc='Training: '):
        optim.zero_grad()
        pred = model(x)
        loss = 0.5 * (torch.mean((pred.permute(2, 0, 1)-img_coarse)**2))  # 256x256
        loss.backward()
        sys.stderr.write(loss.detach().item())
        optim.step()


    plt.imshow(pred.detach().cpu(), cmap='gray')
    plt.savefig('tmp/train_fourfeat_{}_{}.jpg'.format(gridDimensions, scale_factor))

    gridDimensionsTest = np.array([512, 512])
    mgrid_test = utils.MeshGrid(gridDimensionsTest, domain, flatten=False)
    x_test = next(iter(mgrid_test))
    x_test = x_test.cuda()

    def criterion(y1, y2):
        return .5 * torch.mean((y1 - y2) ** 2)
    def psnr(y1, y2):
        return -10 * torch.log10(2.*criterion(y1, y2))

    img_coarse = img_coarse.unsqueeze(0)

    # fourfeat interplation
    pred_test = model(x_test)
    loss_value = criterion(pred_test.detach().permute(2, 0, 1), img)
    psnr_value = psnr(pred_test.detach().permute(2, 0, 1), img)
    title = 'test_{}_{}_loss{:.3f}_psnr{:.3f}.jpg'.format('fourfeat', gridDimensionsTest, loss_value, psnr_value)
    plt.imshow(pred_test.detach().cpu(), cmap='gray')
    plt.title(title)
    plt.savefig('tmp/test_fourfeat_{}_{}.jpg'.format(gridDimensionsTest, scale_factor))

    interpolation_modes = ['nearest', 'bilinear', 'bicubic', 'area']

    for in_mode in interpolation_modes:
        interpolated = functional.interpolate(input=img_coarse, scale_factor=float(scale_factor), mode=in_mode) 
        loss_value = criterion(interpolated[0], img)
        psnr_value = psnr(interpolated[0], img)
        sys.stderr.write('{} interpolation - loss: {}, psnr : {}\n'.format(in_mode, loss_value, psnr_value))

        plt.imshow(interpolated[0].detach().cpu().permute(1, 2, 0), cmap='gray')
        title = 'test_{}_{}_loss{:.3f}_psnr{:.3f}.jpg'.format(in_mode, gridDimensionsTest, loss_value, psnr_value)
        plt.title(title)
        plt.savefig('tmp/test_{}_{}_{}'.format(in_mode, gridDimensionsTest, scale_factor))


def interpolate_coarse_to_fine(coarse_density, problem_path, save, title, size,
                               mode='bilinear', visualize=True, path=None):
    if save:
        # TODO: add ability to accept ``scale_factor`` on top of ``size```

        with open(problem_path, 'r') as j:
            configs = json.loads(j.read())
        # hyperparameters of the problem 
        problem_name = configs['problem_name']
        MATERIAL_PATH = configs['MATERIAL_PATH']
        BC_PATH = configs['BC_PATH']
        orderFEM = configs['orderFEM']
        domainCorners = configs['domainCorners']
        gridDimensions = configs['gridDimensions']
        E0 = configs['E0']
        Emin = configs['Emin']
        SIMPExponent = configs['SIMPExponent']
        maxVolume = torch.tensor(configs['maxVolume'])
        seed = configs['seed']

        density = coarse_density
        size_interpolation = size
        interpolated = nn.functional.interpolate(density.permute(0, 3, 1, 2),
                                                size=size_interpolation,
                                                mode='bilinear', align_corners=False)
        interpolated = interpolated.permute(0, 2, 3, 1)
        gridDimensions = list(size)
        # gridDimensions = [d * scale_factor_interpolation for d in gridDimensions]

        # solve topopt for interpolated densities in higher resolution and report compliance
        constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
        filters = []
        uniformDensity = maxVolume
        tps = initializeTensorProductSimulator(
            orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
        )                                                                                  
        objective = pyVoxelFEM.ComplianceObjective(tps)                                    
        top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, filters)

        top.setVars(interpolated.detach().cpu().flatten().numpy().astype(np.float64))
        interpolated_objective = top.evaluateObjective()

        sys.stderr.write('bilinear_{} | Compliance after interpolation to {}: {}\n'.format(title,
                                                                                           gridDimensions,
                                                                                           interpolated_objective))

        density_binary = (interpolated > 0.5).float() * 1
        if torch.cuda.is_available():
            density_binary = density_binary.cpu()
        top.setVars(density_binary.detach().cpu().flatten().numpy().astype(np.float64))
        binary_compliance_loss = top.evaluateObjective()

        if visualize:
            if path is None:
                path = ''
            visualizations.density_vis(interpolated, interpolated_objective, gridDimensions,
                                       'bilinear_'+title, True,
                                        visualize, binary_loss=binary_compliance_loss, path=path)
