import sys, os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import visualizations
import utils
import fem
import multires_utils

import time, copy
import argparse, json, psutil, ast
from tqdm import tqdm

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh, parallelism # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore


parser=argparse.ArgumentParser()
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
parser.add_argument('--grid', help='grid dimension in a list of int e.g. "[40, 20, 10]"')
parser.add_argument('--prob', help='problem BCs and other arguments in a JSON file')
parser.add_argument('--v0', help='Volume constraint as a float')
parser.add_argument('--mgl', help='Multigrid solver levels (grid size must be divisible by mgl^2)', default=2)
parser.add_argument('--iter', help='maximum number of iteration as a explicit stopping criterion', default=5000)
parser.add_argument('--optim', help='optimizer used for VoxelFEM ("OC" or "LBFGS")', default='OC')
parser.add_argument('--af', help='adaptive filtering as a list of int (Experimental)', default="[1, 1, 1, 1]")

args=parser.parse_args()
experiment_id = args.jid
sys.stderr.write('Experiment ID: {}\n'.format(experiment_id))

parallelism.set_max_num_tbb_threads(psutil.cpu_count(logical=False))
parallelism.set_gradient_assembly_num_threads(min(psutil.cpu_count(logical=False), 8))

visualize = False
save_density = False

# record full runtime
start_time = time.perf_counter()

problem_path = args.prob  # TODO
gridDimensions = ast.literal_eval(args.grid)  # TODO
maxVolume = float(args.v0)  # TODO
max_iter = int(args.iter)  # TODO
optim = args.optim  # TODO
multigrid_levels = int(args.mgl)  # TODO
if multigrid_levels == 0:
    use_multigrid = False
else:
    use_multigrid = True
adaptive_filtering = ast.literal_eval(args.af)  # TODO
sys.stderr.write('adaptive filtering configs: {}\n'.format(adaptive_filtering))

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/gt/'.format(log_base_path)
log_loss_path = '{}loss/gt/'.format(log_base_path)
log_densities_path = '{}densities/gt/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, log_densities_path, None,
                                                experiment_id=args.jid)
log_image_path = '{}images/gt/{}'.format(log_base_path, append_path)
log_loss_path = '{}loss/gt/{}'.format(log_base_path, append_path)
log_densities_path = '{}densities/gt/{}'.format(log_base_path, append_path)
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
SIMPExponent = configs['SIMPExponent']
if maxVolume is None:
    maxVolume = configs['maxVolume'][0]
else:
    configs['maxVolume'] = [maxVolume]
if adaptive_filtering is None:
    adaptive_filtering = configs['adaptive_filtering'][:-2]  # dont use gaussian smoothing
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))
sys.stderr.write('Multigrid levels: {}\n'.format(multigrid_levels))

# reproducibility
np.random.seed(seed)

# vis title
grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
title = str(experiment_id)+'_voxelfem_'+'optim-'+optim+'_'+grid_title+'_'+str(max_iter)+'_'+problem_name+'_Vol'+str(maxVolume)

# solve
gt_tps, gt_loss, binary_gt_loss, gt_loss_array = fem.ground_truth_topopt(MATERIAL_PATH, BC_PATH, orderFEM, domainCorners,
                                                          gridDimensions, SIMPExponent, maxVolume, use_multigrid=use_multigrid,
                                                          init=None, optimizer=optim, multigrid_levels=multigrid_levels,
                                                          adaptive_filtering=adaptive_filtering, 
                                                          max_iter=max_iter, obj_history=True, title=title,
                                                          log_image_path=log_image_path, log_densities_path=log_densities_path)
sys.stderr.write('Final step, Compliance loss {:.6f}, Binary Compliance loss {:.6f} \n'.format(gt_loss, binary_gt_loss))

# visualization and saving model
visualizations.loss_vis(gt_loss_array, title, visualize, path=log_loss_path, 
                        ylim=np.max(gt_loss_array)+0.1*np.max(gt_loss_array))
utils.save_for_interactive_vis(gt_tps, gridDimensions, title, visualize, path=log_image_path)
utils.save_densities(gt_tps, gridDimensions, title, save_density, False, path=log_densities_path)
execution_time = time.perf_counter() - start_time
sys.stderr.write('\nOverall runtime: {}\n'.format(execution_time))
