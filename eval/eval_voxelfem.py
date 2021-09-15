import torch
import torch.nn as nn

import numpy as np
import os, sys, json

import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import visualizations
import fem

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore

parser=argparse.ArgumentParser()
parser.add_argument('--expp', help='Path to the saved densities (result will be saved in "images" with same path)', 
                    default='4449129/voxelfem_300x100_1500_MbbBeamS88_Vol0.3_F[1, 1, 1, 1]_gt.npy')
args=parser.parse_args()
experiment_path = args.expp
sys.stderr.write('Loading pretrained weight with experiment ID and path: {}\n'.format(experiment_path))

problem_path = 'problems/2d/mbb_beam.json'  # TODO
interpolate = False  # TODO
vector_graphics = None  # TODO
test_resolution = [375, 125]  # TODO
max_resolution = [375, 125]  # TODO

mrconfprint = 'Testing saved densities in: {} \n'.format(test_resolution)
mrconfprint += 'Interpolation: {} to: {}\n'.format(interpolate, max_resolution)
sys.stderr.write(mrconfprint)

density_path = 'logs/densities/gt/{}'.format(experiment_path)
images_path = density_path.replace('densities', 'images')
images_path = images_path[:images_path.rfind('/')+1]

####################################################################
# images_path = 'tmp/'
####################################################################

density = np.load(density_path)
density = -torch.from_numpy(density.transpose(1, 0)).float().unsqueeze(0).unsqueeze(0)
if interpolate:
    test_resolution = max_resolution
    density = nn.functional.interpolate(density, test_resolution, mode='bilinear', align_corners=True)
binary_density = (density > 0.5) * 1.
sys.stderr.write('Densities loaded.\n')

# hyperparameters of the problem 
with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
gridDimensions = configs['gridDimensions']
E0 = configs['E0']
Emin = configs['Emin']
SIMPExponent = configs['SIMPExponent']
maxVolume = configs['maxVolume'][0]
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
)                                                                                  
objective = pyVoxelFEM.ComplianceObjective(tps)                                    
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, [])
voxelfem_engine = fem.VoxelFEMFunction.apply
compliance_loss = voxelfem_engine(density.flatten(), top)
sys.stderr.write('Compliance loss: {} and volume constraint={}\n'.format(compliance_loss, density.mean()))
binary_compliance_loss = voxelfem_engine(binary_density.flatten(), top)
sys.stderr.write('Binary Compliance loss: {} and binary volume constraint={}\n'.format(binary_compliance_loss, binary_density.mean()))

# visualization
grid_title = ''.join(str(i)+'x' for i in test_resolution)[:-1]
expp_title = args.expp[args.expp.rfind('/')+1:-3]
title = 'testDensities-{}_{}_Vol{}_intpol-{}_'.format(expp_title, grid_title, maxVolume, interpolate)
visualizations.density_vis(density, compliance_loss, test_resolution, title, False, True, True,
                           binary_loss=binary_compliance_loss, path=images_path)
sys.stderr.write('Test image saved to: {}\n'.format(images_path))

if vector_graphics is not None:
    plt.imshow(-density.squeeze(0).squeeze(0).T, cmap='gray')
    plt.savefig('tmp/{}.{}'.format(title, vector_graphics), bbox_inches='tight')
