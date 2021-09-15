import numpy as np

from functools import partial

import sys, os, time

import filtering
import utils

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore

import torch
import torch.autograd as autograd


def ground_truth_topopt(MATERIAL_PATH, BC_PATH, orderFEM, 
                        domainCorners, gridDimensions, SIMPExponent,
                        maxVolume, optimizer, multigrid_levels,
                        use_multigrid=True, adaptive_filtering=[1, 1, 1, 1],
                        max_iter=100, init=None, obj_history=False, **kwargs):

    # Visualization/saving
    title=kwargs['title']
    log_image_path = kwargs['log_image_path']
    log_densities_path = kwargs['log_densities_path']
    
    E0 = 1
    Emin = 1e-4  # TODO

    constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
    filters = [
        pyVoxelFEM.SmoothingFilter(),
        pyVoxelFEM.ProjectionFilter(),
    ]
    uniformDensity = maxVolume
    tps = initializeTensorProductSimulator(
        orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)
    if use_multigrid:
        objective = pyVoxelFEM.MultigridComplianceObjective(tps.multigridSolver(multigrid_levels))
    else:
        objective = pyVoxelFEM.ComplianceObjective(tps)                                    
    top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, filters) 
    nonLinearProblem, problemObj = initializeIpoptProblem(top)
    
    # for filt in filters:
    #     if isinstance(filt, pyVoxelFEM.SmoothingFilter):
    #         filt.radius = 5

    # add adaptive filtering
    if adaptive_filtering is not None:
        problemObj.beta_interval, problemObj.beta_scaler, problemObj.radius_interval, problemObj.radius_scaler = adaptive_filtering
    
    if init is None:
        x0 = tps.getDensities()
    else:
        init = init.numpy().astype(np.float64)
        top.setVars(init.flatten())
        x0 = tps.getDensities()
    
    if use_multigrid:
        # Configure multigrid objective
        objective.tol = 1e-4  # TODO
        objective.mgIterations = 2
        objective.fullMultigrid = True

    if optimizer == 'OC':        
        oco = pyVoxelFEM.OCOptimizer(top)
        top.setVars(tps.getDensities())
        ckp_step = max_iter // 10
        iter_start_time = 0
        for idx in range(max_iter):
            iter_time = time.perf_counter() - iter_start_time
            objective_value = 2.0 * top.evaluateObjective()
            problemObj.history.objective.append(objective_value)
            sys.stderr.write('Total Steps: {:d}, Runtime: {:.1f}, Compliance loss {:.6f}\n'.format(idx, iter_time, objective_value))
            if (idx+1) % ckp_step == 0:
                title_ = '{}_iter{}'.format(title, idx)
                utils.save_for_interactive_vis(tps, gridDimensions, title_, True, path=log_image_path)
                utils.save_densities(tps, gridDimensions, title_, True, False, path=log_densities_path)
            iter_start_time = time.perf_counter()
            oco.step()

    elif optimizer == 'LBFGS':
        nonLinearProblem.addOption('print_level', 0)
        nonLinearProblem.addOption(b'sb', b'yes')
        nonLinearProblem.addOption('max_iter', max_iter)
        nonLinearProblem.addOption('tol', 1e-7)

        x0, _ = nonLinearProblem.solve(x0)
    else:
        raise ValueError('Optimizer {} is unknown or not implemented.'.format(optimizer))
    x0 = tps.getDensities()
    binary_objective = utils.compute_binary_compliance_loss(density=x0, loss_engine=None, top=top)

    if obj_history is False:
        return (tps if len(orderFEM) == 3 else tps.getDensities(), 
                2.0 * top.evaluateObjective(), binary_objective)
    else:
        return (tps if len(orderFEM) == 3 else tps.getDensities(),
                2.0 * top.evaluateObjective(), binary_objective, problemObj.history.objective)


class VoxelFEMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, densities: torch.Tensor, top):  # type: ignore
        """

        :param ctx: pytorch context manager
        :param densities: predicted densities (xPhys)
        :param top: topology optimization object instantiated from ``pyVoxelFEM.TopologyOptimizationProblem``
        """  

        # nlp.solve for 0 iterations = top.__objective(densities) 
        ## where densities were updated using ``top.setVars(densities)``
        top.setVars(densities.numpy().astype(np.float64))
        output_objective = 2.0 * top.evaluateObjective()
        
        # already accumulated from ``top.setVars(densities)``
        output_gradient = top.evaluateObjectiveGradient().astype(np.float32)
        output_gradient = torch.from_numpy(output_gradient)
        ctx.save_for_backward(output_gradient) 

        return torch.tensor(output_objective).float()

    @staticmethod
    def backward(ctx, grad_output):
        output_gradient = ctx.saved_tensors[0]
        return (output_gradient * grad_output), None


class FindRootFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, y, average, lower_bound, upper_bound, tolerance=1e-12, max_iterations=128, projection=None):
        """
        Implicitly solve f(x,y)=0 for y(x) using binary search where f = lambda x, y: projection(x + y).mean() - average
        Assumes that y is a scalar and f(x,y) is monotonic in y.

        :param ctx: pytorch context manager
        """

        step = 0
        while (step < max_iterations) and (upper_bound - lower_bound >= tolerance):
            y = 0.5 * (lower_bound + upper_bound)
            if (projection(x + y).mean() - average) > 0:
                upper_bound = y
            else:
                lower_bound = y
            step = step + 1
        
        y = 0.5 * (lower_bound + upper_bound)

        if torch.cuda.is_available():
            y = y.clone().detach().cuda().requires_grad_(True)
            x = x.clone().detach().cuda().requires_grad_(True)

            average = torch.tensor([average]).cuda()
        else:
            y = y.clone().detach().requires_grad_(True)
            x = x.clone().detach().requires_grad_(True)

            average = torch.tensor([average]).float()

        # pytorch enforces no grad in forward and backward
        with torch.set_grad_enabled(True):
            f = projection(x + y).mean() - average
        dfdx = autograd.grad(f, x)[0].detach()
        with torch.set_grad_enabled(True):  # TODO: currently retain_graph does not work
            f = projection(x + y).mean() - average
        dfdy = autograd.grad(f, y)[0].detach()

        ctx.save_for_backward(x.detach(), y.detach(), average, dfdx, dfdy)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, average, dfdx, dfdy = ctx.saved_tensors

        # we are looking for 
        ## dfdx = -1
        ## dfdy = 2*y where y is from forward = 1.4142
        ## grad = - dfdx / dfdy
        return -dfdx / dfdy * grad_output, None, None, None, None, None, None, None


# instantiate above class for following method 
find_root = FindRootFunction.apply


def physical_density(x, maxVolume):
    """
    Computes physical densities x from predicted logit densities x_hat (input ``x``)

    :param x: Input density logits (unconstrainted)
    :param maxVolume: Maximum amount of volume
    :return: Constrainted ``x`` which satisfies volume constraint given by ``maxVolume`` with same shape as ``x``
    """

    x = sigmoid_with_constrained_mean(x, maxVolume)
    return x


def sigmoid_with_constrained_mean(x, average, projection=torch.sigmoid):
    """
    Satisfy reduction constraint by pushing average of input x toward input argument ``average``
    In this method, sigmoid of input is satisfied.

    :param x: Constrained input tensor 
    :param average: The constaint value
    :param projection: Function as the projection of values to binary (``torch.sigmoid`` here)
    :return: Satisfied version of input ``x``
    """

    # f = lambda x, y: torch.sigmoid(x + y).mean() - average
    lower_bound = logit(average) - torch.max(x)
    upper_bound = logit(average) - torch.min(x)
    y = 0.5 * (lower_bound + upper_bound)
    b = find_root(x, y, average, lower_bound, upper_bound, 1e-12, 128, projection)
    return projection(x + b)


def projection_filter_with_constrained_mean(x, average, projection=None):
    """
    Satisfy reduction constraint by pushing average of input x toward input argument ``average``
    In this method, sigmoid of input is satisfied.

    :param x: Constrained input tensor 
    :param average: The constaint value
    :param beta: Beta hyperparameter of projection filter (higher ``beta``, closer to step function)
    :param projection: Function as the projection of values to binary (``filtering.ProjectionFilter()`` here)
        use ``ProjectionFilter(beta, normalized=False)`` to prevent positive definitness error
    :return: Satisfied version of input ``x``
    """

    if projection is None:
        filtering.ProjectionFilter(beta=1)

    lower_bound = logit(average) - torch.max(x)
    upper_bound = logit(average) - torch.min(x)
    y = 0.5 * (lower_bound + upper_bound)
    b = find_root(x, y, average, lower_bound, upper_bound, 1e-12, 128, projection)
    return projection(x + b)


def logit(p):
    p = torch.clamp(p, 0, 1)
    return torch.log(p) - torch.log1p(-p)


# wrapper around all volume constraint satisfaction methods
def satisfy_volume_constraint(density, max_volume, compliance_loss=None, 
                              mode='constrained_sigmoid', scaler_mode='clip', constant=500., **kwargs):
    """
    Soft/Hard methods to satisfy volume constraint during training

    :return: A tuple of (density, volume_loss)
    """
    
    # even though density is now on CPU, but because of following two line, operation will happen on GPU (does not matter actually)
    current_volume = torch.zeros_like(max_volume).fill_(torch.mean(density))
    zero_tensor = torch.zeros_like(max_volume)

    if mode == 'constrained_sigmoid':
        # google method
        return sigmoid_with_constrained_mean(x=density, average=max_volume, projection=torch.sigmoid)

    elif mode == 'constrained_projection':
        # default voxelfem binarization method: recommended (even in case of default values)
        projection = kwargs['projection'] if 'projection' in kwargs else None
        density = projection_filter_with_constrained_mean(x=density, average=max_volume, projection=projection)
        return density
        
    elif mode == 'add_mean':
        # enforces volume constaint **equality** by computing difference between current volume and desired volume
        volume_loss = torch.abs(current_volume - max_volume)
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler

    elif mode == 'one_sided_max':
        # enforces volume constraint **inequality** ``max(V - V_max)^2``
        volume_loss = torch.maximum(current_volume - max_volume, zero_tensor) ** 2
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler

    elif mode == 'maxed_barrier':
        # enforces volume constraint **inequality** ``max(-log(1 + V_max + eps - x), 0)``
        eps = 1e-7
        volume_loss = torch.maximum(-torch.log(1 + max_volume + eps - current_volume), zero_tensor)
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler

    elif mode == 'thresholded_barrier':
        # enforces volume constraint **inequality** ``min(log(a / (V_max - V), 0)^2`` where ``a`` is activation threshold
        eps = 1e-7
        a = 1 + max_volume + eps - current_volume if current_volume <= max_volume else 1.
        volume_loss = torch.log(a / (1 + max_volume + eps - current_volume)) ** 2
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler


def compute_volume_loss_scaler(compliance_loss, volume_loss, mode='clip', constant=500.):
    """
    As volume constraint loss is much smaller in the begining, we add a scaler as a weight to increase/decrease its value


    :param compliance_loss: Compliance loss for given density
    :param volume_loss: Volume loss for given density

    :param mode: A heuristic that changes ``scaler``
    :param constant: Used for ``mode='clip'``

    :return: A scaler as the weight for `volume_loss` in weighted sum `compliance_loss + scaler * volume_loss`
    """
    with torch.no_grad():
        scaler = compliance_loss / volume_loss

        if mode == 'clip':
            if scaler >= constant:
                scaler = torch.clamp_max(scaler, max=constant)
                return scaler
            else:
                return scaler
        elif mode == 'equalize':
            return scaler


def type_of_volume_constaint_satisfier(mode):
    """
    Says mode is hard or not i.e. change the density directly or add a loss term respectively

    """
    if mode == 'constrained_sigmoid': return True
    elif mode == 'constrained_projection': return True
    elif mode == 'add_mean': return False
    elif mode == 'one_sided_max': return False
    elif mode == 'maxed_barrier': return False
    elif mode == 'thresholded_barrier': return False
    else: raise ValueError('The mode "{}" does not exist'.format(mode))


def homogeneous_init(model, constant):
    """
    Ensures the first output of model is a homogeneous field by zeroing out weights and initializing bias
      with a `constant`. This function is inplace.
    
    :param model: A `Module` model
    :param constant: A float scalar
    :return: None
    """

    def apply_homogeneous_init(m, constant):
            """
            Zero outs weights of last layer and initialize biases with `constant` value
            Used to ensure first output of neural network is homogeneous density field.

            :param m: Module m (rec: use module.apply(this method))
            """
            classname = m.__class__.__name__
            if (classname.find('Linear') != -1):
                if ((m.weight.shape.__contains__(1)) or m.weight.shape.__contains__(2)):
                    torch.nn.init.normal_(m.weight, 0.0, 0.0001)
                    torch.nn.init.constant_(m.bias, constant)

    model.apply(partial(apply_homogeneous_init, constant=constant))
    sys.stderr.write('Homogenization has been applied on model with constant value: {}\n'.format(constant))
