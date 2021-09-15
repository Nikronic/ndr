import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
from fem import VoxelFEMFunction
from utils import MeshGrid
import fem
import utils
import filtering
import multires_utils
import visualizations

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
import json, time, ast
import argparse, psutil
import itertools
import copy

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh, parallelism  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore


parser=argparse.ArgumentParser()
# VoxelFEM conf
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
parser.add_argument('--grid', help='grid dimension in a list of int e.g. "[40, 20, 10]"')
parser.add_argument('--prob', help='problem BCs and other arguments in a JSON file')
parser.add_argument('--v0', help='Volume constraint as a float')
parser.add_argument('--mgl', help='Multigrid solver levels (grid size must be divisible by mgl^2)', default=2)

# EDRN conf
parser.add_argument('--vcs', help='volume constraint satisfier', default='maxed_barrier')
parser.add_argument('--checkpoint', help='path to previous pytorch checkpoint (e.g. "logs/weights/ff/4932098/4932098_iter2996.pt"). In case of "None", training will be started all over again.')
parser.add_argument('--es', help='Fourier features size (embedding size)', default=1024)
parser.add_argument('--nn', help='number of neurons (width size)', default=512)
parser.add_argument('--nl', help='number of hidden layers (depth size)', default=4)
parser.add_argument('--lr', help='learning rate of Adam optim', default=3e-4)
parser.add_argument('--iter', help='maximum number of iteration as a explicit stopping criterion', default=5000)
parser.add_argument('--cs', help='checkpoint size (an instance will be saved by iter/cs)', default=100)
parser.add_argument('--sigma', help='Scale (sigma) of the Fourier features Gaussian distribution')


args=parser.parse_args()

experiment_id = args.jid
sys.stderr.write('Experiment ID: {}\n'.format(experiment_id))

# debug codes
def GPU_OFF():
    return False
torch.cuda.is_available = GPU_OFF
if experiment_id is None:
    sys.stderr.write = print

# PyTorch related global variables
torch.autograd.set_detect_anomaly(False)
parallelism.set_max_num_tbb_threads(psutil.cpu_count(logical=False))

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_density = True
fixed_scale = True

# record runtime
start_time = time.perf_counter()
# cantilever_flexion
problem_path = args.prob  # TODO
checkpoint_path = args.checkpoint  # TODO
gridDimensions = ast.literal_eval(args.grid)  # TODO
max_resolution = None  # TODO

# multires hyperparameters
volume_constraint_satisfier = args.vcs
# using filtering as post processing after each iteration (e.g. does not affect if combined with constraint satisfaction)
## (0<...<1, _, True|False) means (no update, _, usage) filters respectively
adaptive_filtering_configs = {}
adaptive_filtering_configs['projection_filter'] = False
adaptive_filtering_configs['beta_init'] = 1
adaptive_filtering_configs['beta_interval'] = 0.1
adaptive_filtering_configs['beta_scaler'] = -1
adaptive_filtering_configs['smoothing_filter'] = False
adaptive_filtering_configs['radius_init'] = 1
adaptive_filtering_configs['radius_interval'] = 0.1
adaptive_filtering_configs['radius_scaler'] = -1
adaptive_filtering_configs['gaussian_filter'] = False
adaptive_filtering_configs['sigma_init'] = 1
adaptive_filtering_configs['sigma_interval'] = 0.1
adaptive_filtering_configs['sigma_scaler'] = -1

is_volume_constraint_satisfier_hard = fem.type_of_volume_constaint_satisfier(mode=volume_constraint_satisfier)
embedding_size = int(args.es)  # TODO
n_neurons = int(args.nn)  # TODO
n_layers  = int(args.nl)  # TODO
multigrid_levels = int(args.mgl)  # TODO
learning_rate = float(args.lr)  # TODO
maxVolume = float(args.v0)  # TODO
checkpoint_size = int(args.cs)  # TODO
weight_decay = 0.0
use_scheduler = None
epoch_mode = 'constant'

# hyper parameter of positional encoding in NeRF
scale = [float(args.sigma)]  # TODO
use_multigrid = True
SIMPExponent = 3
resolutions = multires_utils.prepare_resolutions(interval=0, start=0, end=1, order='ftc', repeat_res=1)[:-1]  
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions),
                                                 start=800, end=1500, 
                                                 mode=epoch_mode, constant_value=int(args.iter))  # TODO
mrconfprint = 'epoch mode: {}, '.format(epoch_mode)
mrconfprint += 'adaptive filtering configs: {} \n'.format(adaptive_filtering_configs)
mrconfprint += 'Volume constraint satisfier: {} (hard: {})\n'.format(volume_constraint_satisfier,
                                                                     is_volume_constraint_satisfier_hard)
sys.stderr.write(mrconfprint)
sys.stderr.write('scale: {}, fourier embedding size: {}\n'.format(scale, embedding_size))
sys.stderr.write('Multigrid solver coarse levels: {}\n'.format(multigrid_levels))

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/ff/'.format(log_base_path)
log_loss_path =  '{}loss/ff/'.format(log_base_path)
log_weights_path = '{}weights/ff/'.format(log_base_path)
log_densities_path = '{}densities/ff/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, log_densities_path,
                                                log_weights_path, None, experiment_id=args.jid)
log_image_path = '{}images/ff/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/ff/{}'.format(log_base_path, append_path)
log_weights_path =  '{}weights/ff/{}'.format(log_base_path, append_path)
log_densities_path = '{}densities/ff/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))


with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

# hyperparameters of the problem 
problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
if gridDimensions is None:
    gridDimensions = configs['gridDimensions']
else:
    configs['gridDimensions'] = gridDimensions
E0 = configs['E0']
Emin = configs['Emin']
if SIMPExponent is None:
    SIMPExponent = configs['SIMPExponent']
else:
    configs['SIMPExponent'] = SIMPExponent
if maxVolume is None:
    maxVolume = torch.tensor(configs['maxVolume'])
else:
    configs['maxVolume'] = [maxVolume]
    maxVolume = torch.tensor(configs['maxVolume'])
if adaptive_filtering_configs is None:
    adaptive_filtering_configs = configs['adaptive_filtering']
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

gridDimensions_ = copy.deepcopy(gridDimensions)
if max_resolution is None:
    max_resolution = gridDimensions

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()

domain = np.array([[0., 1.], [0., 1.], [0., 1.]])
sys.stderr.write('Domain: {}\n'.format(domain))

# deep learning modules
if is_volume_constraint_satisfier_hard:
    nerf_model = MLP(in_features=3, out_features=1, n_neurons=n_neurons, n_layers=n_layers, embedding_size=embedding_size,
                     scale=scale[0], hidden_act=nn.ReLU(), output_act=None)
else:
    nerf_model = MLP(in_features=3, out_features=1, n_neurons=n_neurons, n_layers=n_layers, embedding_size=embedding_size,
                     scale=scale[0], hidden_act=nn.ReLU(), output_act=nn.Sigmoid())
model = nerf_model
if torch.cuda.is_available():
    model.cuda()

# apply homogenization
fem.homogeneous_init(model=model, constant=maxVolume.item())
sys.stderr.write('Deep learning model config: {}\n'.format(model))

# filtering
projection_filter = filtering.ProjectionFilter(beta=adaptive_filtering_configs['beta_init'], normalized=True)
smoothing_filter = filtering.SmoothingFilter(radius=adaptive_filtering_configs['radius_init'])
gauss_smoothing_filter = filtering.GaussianSmoothingFilter(sigma=adaptive_filtering_configs['sigma_init'])
filters = [projection_filter, smoothing_filter, gauss_smoothing_filter]

# optim
optim = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=weight_decay)

# loading checkpoint if any path has been provided
init_step = 0
if checkpoint_path is not None:
    init_step = utils.load_weights(model, optim, checkpoint_path)

# reduce on plateau
scheduler = None
if use_scheduler == 'reduce_on_plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=10)
if use_scheduler == 'multi_step_lr':
    milestones_step = 100
    milestones = [i*milestones_step for i in range(1, 4)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=milestones, gamma=0.3)
sys.stderr.write('DL optim: {}, LR scheduler: {}\n'.format(optim, scheduler))
sys.stderr.write('L2 Regularization: {}\n'.format(weight_decay))

# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
compliance_loss_array = []

for idx, res in enumerate(resolutions):
    
    model.train()

    gridDimensions = tuple(np.array(gridDimensions_) + res * np.array(domainCorners[1]))
    sys.stderr.write('New resolution within multires loop: {}\n'.format(gridDimensions))

    if torch.cuda.is_available():
        maxVolume = maxVolume.cuda()

    # deep learning modules
    dataset = MeshGrid(sidelen=gridDimensions, domain=domain, flatten=False)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    model_input = next(iter(dataloader))

    # we dont want to update input in nerf so dont enable grads here
    if torch.cuda.is_available():
        model_input = model_input.cuda()

    # topopt (via VoxelFEM-Optimization-Problem)
    constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
    uniformDensity = maxVolume
    tps = initializeTensorProductSimulator(
        orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)                                                                                  
    if use_multigrid:
        objective = pyVoxelFEM.MultigridComplianceObjective(tps.multigridSolver(multigrid_levels))
        objective.tol = 1e-4  # TODO
        objective.mgIterations = 2
        objective.fullMultigrid = True
    else:
        objective = pyVoxelFEM.ComplianceObjective(tps)
    top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, []) 

    # instantiate autograd.Function for VoxelFEM engine
    voxelfem_engine = VoxelFEMFunction.apply

    # reset adaptive filtering
    filtering.reset_adaptive_filtering(filters=filters, configs=adaptive_filtering_configs)

    # save loss values for plotting
    compliance_loss_array_res = []

    ckp_step = epoch_sizes[idx] // checkpoint_size
    # training of xPhys
    for step in tqdm(range(init_step, epoch_sizes[idx]), desc='Training: '):

        def closure():
            optim.zero_grad()

            # aka x
            density = model(model_input)
            density = density.view(gridDimensions)

            # aka xPhys
            if is_volume_constraint_satisfier_hard:
                density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                        mode=volume_constraint_satisfier, projection=projection_filter)
            else: 
                density = torch.clamp(density, min=0., max=1.)
            
            # adaptive filtering
            if adaptive_filtering_configs is not None:
                density = filtering.apply_filters_group(x=density, filters=filters, configs=adaptive_filtering_configs)
                filtering.update_adaptive_filtering(iteration=step, filters=filters, configs=adaptive_filtering_configs)

            # compliance for predicted xPhys
            if torch.cuda.is_available():
                density = density.cpu()
            compliance_loss = voxelfem_engine(density.flatten(), top)
            if torch.cuda.is_available():
                compliance_loss.cuda()

            global actual_steps
            actual_steps += 1

            # for 'soft' volume constraint 
            # TODO: #62 make it adaptive, first use small values then after 100, 200 iterations, use the larger recommended value
            if not is_volume_constraint_satisfier_hard:
                volume_loss = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume,
                                                            compliance_loss=compliance_loss, 
                                                            scaler_mode='clip', constant=1500,
                                                            mode=volume_constraint_satisfier)
                sys.stderr.write('\n{} with mode: {} with constant: {} -> v-loss={}\n'.format(volume_constraint_satisfier,
                                                                            'clip', 1500, volume_loss.clone().detach().item()))
                compliance_loss = compliance_loss + volume_loss

            compliance_loss.backward()

            # save loss values for plotting
            compliance_loss = compliance_loss.detach().item()
            compliance_loss_array_res.append(compliance_loss)
            sys.stderr.write('Total Steps: {:d}, Resolution Steps: {:d}, Compliance loss {:.6f}\n'.format(actual_steps,
                                                                                                    step, compliance_loss))
            return compliance_loss

        optim.step(closure)

        # reduce LR if no reach plateau
        if use_scheduler is not None:
            if use_scheduler == 'reduce_lr_on_plateau':
                scheduler.step(compliance_loss_array_res[-1])
            else:
                scheduler.step()
        
        with torch.no_grad():
            if (step+1) % ckp_step == 0:
                title_ = append_path[:-1] if args.jid is None else args.jid
                title_ = '{}_iter{}'.format(title_, step)
                utils.save_weights(model=model, step=step, optim=optim,
                                title=title_, save=save_model, path=log_weights_path)
                
                density = model(model_input)
                if is_volume_constraint_satisfier_hard:
                    density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                            mode=volume_constraint_satisfier, projection=projection_filter)
                else: 
                    density = torch.clamp(density, min=0., max=1.)
                utils.save_for_interactive_vis(density, gridDimensions, title_, visualize, path=log_image_path)
                utils.save_densities(tps, gridDimensions, title_, save_density, True, path=log_densities_path)

    compliance_loss_array.extend(compliance_loss_array_res)

    # test model with for res idx
    with torch.no_grad():
        density = model(model_input)
        if is_volume_constraint_satisfier_hard:
            density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                    mode=volume_constraint_satisfier, projection=projection_filter)
        else: 
            density = torch.clamp(density, min=0., max=1.)

        # loss of conversion to binary by thresholding
        binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                    loss_engine=voxelfem_engine)

        # visualization and saving model
        grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
        maxVolume_np = maxVolume.detach().cpu().numpy()
        title = str(experiment_id)+'_FF(HC'+str(is_volume_constraint_satisfier_hard)+')_s'+str(scale)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)
        title +=  problem_name+'_Vol'+str(maxVolume_np)
        title = visualizations.loss_vis(compliance_loss_array_res, title, True, path=log_loss_path,
                                        ylim=np.max(compliance_loss_array_res) if np.max(compliance_loss_array_res) < 1000 else 1000)

# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# now query for max resolution after training finished
test_resolution = max_resolution
dataset = MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)                                                                                  
if use_multigrid:
    objective = pyVoxelFEM.MultigridComplianceObjective(tps.multigridSolver(multigrid_levels))
    objective.tol = 1e-4
    objective.mgIterations = 2
    objective.fullMultigrid = True
else:
    objective = pyVoxelFEM.ComplianceObjective(tps)
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, []) 

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()
voxelfem_engine = VoxelFEMFunction.apply

with torch.no_grad():
    model.eval()
    density = model(model_input)
    if is_volume_constraint_satisfier_hard:
        density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                mode=volume_constraint_satisfier, projection=projection_filter)
    else:
        density = torch.clamp(density, min=0., max=1.)

# loss of conversion to binary by thresholding
binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                              loss_engine=voxelfem_engine)
if torch.cuda.is_available():
    density = density.cpu()
compliance_loss = voxelfem_engine(density.flatten(), top)
maxVolume_np = maxVolume.detach().cpu().numpy()
grid_title = ''.join(str(i)+'x' for i in test_resolution)[:-1]
title = str(experiment_id)+'_FF(HC'+str(is_volume_constraint_satisfier_hard)+'_test)_s'+str(scale)+'_'+grid_title+'_'+str(actual_steps)
title += problem_name+'_Vol'+str(maxVolume_np)
utils.save_densities(tps, gridDimensions, title, save_density, True, path=log_densities_path)
utils.save_for_interactive_vis(density, gridDimensions, title, visualize, path=log_image_path)
title = title.replace('test', 'overall')
compliance_loss_array.append(compliance_loss)
title = visualizations.loss_vis(compliance_loss_array, title, True, path=log_loss_path,
                                ylim=np.max(compliance_loss_array) if np.max(compliance_loss_array) < 1000 else 1000)
utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)
