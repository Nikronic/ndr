import torch

import numpy as np
import os, sys


sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore


def prepare_resolutions(interval=5, start=0, end=10, order='ctf', repeat_res=1):
    """
    Prepare an array of changes in resolutions given interval and range. 
    Resoluions to train with - all of them follow same ratio using ``domainCorners`` in ``top``.

    :param interval: Interval between each sampled resolution
    :param start: Smallest resolution
    :param end: Largest resolution
    :param order: 1. 'ctf': coarse-to-fine (increasing order)
                  2. 'ftc': fine-to-coarse (decreasing order)
                  3. 'bidirectional': Zigzag between coarse and fine [WIP]
                  4. 'random': A random order of resolutions
                  5. 'manual: An array of indices providing the order [WIP]
    :param repeat_res: Repeat each resolution n times while respecting ``order``
    :return: An array of changes that need to be added to a base resolution
    """

    resolutions = np.arange(start=start, stop=end) * interval
    resolutions = np.concatenate(tuple([resolutions] * repeat_res))
    
    if order == 'ctf':
        return np.concatenate([resolutions, np.array([resolutions[-1]])])
    elif order == 'ftc':
        return -np.concatenate([resolutions, np.array([0])])
    elif order == 'bidirectional':
        raise NotImplementedError('Not yet!')
    elif order == 'random':
        np.random.shuffle(resolutions)
        return resolutions
    elif order == 'manual':
        raise NotImplementedError('Not yet!')
    else:
        raise NotImplementedError('Mode does not exist or has not been implemented yet!')


def prepare_epoch_sizes(n_resolutions, start=500, end=2000, mode='constant', constant_value=1500):
    """
    Prepare an array of number of iterations (epochs) for each resolution

    :param n_resolutions: Number of resolutions in solver
    :param start: Smallest resolution - ignored when ``mode='constant'``
    :param end: Largest resolution - ignored when ``mode='constant'``
    :param constant_value: Constant iteration number - only used if ``mode='constant'``
    :param order: 1. 'constant': Constant number of iterations
                  2. 'linear_inc': Linearly increasing
                  3. 'linear_dec': Linearly decreasing
                  4. 'linear_abs': Linearly decrease, then increase to the same starting value
                  4. 'random': Uniformly random (does not make sense I know!)
    
    """

    if mode == 'constant':
        return [constant_value] * n_resolutions
    elif mode == 'linear_inc':
        return list(np.linspace(start=start, stop=end, num=n_resolutions, dtype=np.int))
    elif mode == 'linear_dec':
        return list(np.linspace(start=end, stop=start, num=n_resolutions, dtype=np.int))
    elif mode == 'linear_abs':
        dec = list(np.linspace(start=end, stop=start, num=n_resolutions, dtype=np.int))
        inc = list(np.linspace(start=start, stop=end, num=n_resolutions, dtype=np.int))
        if n_resolutions % 2 != 0:
            return list(np.concatenate([dec[::2], inc[:-2:2]]))
        else:
            return list(np.concatenate([dec[::2], inc[::2]]))
    elif mode == 'random':
        return list(np.random.uniform(low=start, high=end, size=(n_resolutions, )).astype(np.int))
    else:
        raise NotImplementedError('Mode does not exist or has not been implemented yet!')


def mkdir_multires_exp(base_image_path, base_loss_path, base_densities_path=None, base_weights_path=None,
                        base_slurm_path=None, experiment_id=None):
    """
    Create exp{some_string_counter} directory for each run

    :param base_image_path: Path to images (densities) dir
    :param base_loss_path: Path to losses (compliances) dir
    :param base_densities_path: Path to densities (saved as numpy) dir
    :param base_slurm_path: Path to slurm logs dir - currently IGNORED!
    :return: New path to exp{id} or slurm_id path
    """

    # TODO: add .npy densities

    if experiment_id is None:
        i = 1
        flag = True
        while flag:
            path = '{}exp{}'.format(base_image_path, i)
            if base_loss_path is not None:
                path_loss = '{}exp{}'.format(base_loss_path, i)
            if base_densities_path is not None:
                path_densities = '{}exp{}'.format(base_densities_path, i)
            if base_weights_path is not None:
                path_weights = '{}exp{}'.format(base_weights_path, i)
            
            # path_slurm = '{}exp{}'.format(base_slurm_path, i)
            if os.path.isdir(path):
                i += 1
            else:
                os.mkdir(path)
                if base_loss_path is not None:
                    os.mkdir(path_loss)
                if base_densities_path is not None:
                    os.mkdir(path_densities)
                if base_weights_path is not None:
                    os.mkdir(path_weights)

                flag = False
                return 'exp{}/'.format(i)
    else:
        path = '{}/{}/'.format(base_image_path, experiment_id)
        os.mkdir(path)
        if base_loss_path is not None:
            path_loss = '{}/{}/'.format(base_loss_path, experiment_id)
            os.mkdir(path_loss)
        if base_densities_path is not None:
            path_densities = '{}/{}/'.format(base_densities_path, experiment_id)
            os.mkdir(path_densities)
        if base_weights_path is not None:
            path_weights = '{}/{}/'.format(base_weights_path, experiment_id)
            os.mkdir(path_weights)
        
        # path_slurm = '{}/{}/'.format(base_slurm_path, experiment_id)
        # os.mkdir(path_slurm)
        return '{}/'.format(experiment_id)


def forget_weights(model, rate, mode='orthogonal', mean=0, std=0.1, lb=-1., ub=1., 
                   n_neurons=256, embedding_size=256, constant_value=1e-2):
    """
    Forget weights of a network (all layers) given the percentage ``rate`` 
      and reinitialize them given distribution ``mode``

    :param  rate: Percentage of weights to be reinitilized
    :param mode: 1. 'orthogonal': Sampling reinitialized weights from linear orthogonal 
                    (needs ``n_neurons`` and ``embedding_size``)
                 2. 'normal': Sampling reinitialized weights from normal (needs ``mean`` and ``std``)
                 3. 'uniform': Sampling reinitialized weights from uniform (needs ``lb``, ``ub``)
                 4. 'constant': Sampling reinitialized weights from a ``constant_value``
    """

    new_state_dict = {}  # type: ignore

    for k in model.state_dict().keys():
        weights = model.state_dict()[k]
        mask = torch.rand_like(weights) > rate
        mask_values = torch.empty(size=(int(mask.sum()), ))
        if torch.cuda.is_available():
            mask_values = mask_values.cuda()
        if len(mask.shape) > 1:  # weights
            if mode == 'orthogonal':
                gain = 1.0 * np.sqrt(max(n_neurons / embedding_size, 1))
                torch.nn.init.orthogonal_(mask_values.unsqueeze(0), gain=gain)
            elif mode == 'normal':
                torch.nn.init.normal_(mask_values, mean=mean, std=std)
            elif mode == 'uniform':
                torch.nn.init.uniform_(mask_values, a=lb, b=ub)
            elif mode == 'constant':
                torch.nn.init.constant_(mask_values, val=constant_value)
            else:
                raise NotImplementedError('Mode {} is invalid or has not been implemented yet!'.format(mode))
        else:  # biases
            torch.nn.init.constant_(mask_values, 0.0)
        weights[mask] = mask_values
        new_state_dict[k] = weights
    model.load_state_dict(new_state_dict)


def forget_activations(model, model_input, mode='dropout', rate=0.8):
    """
    Applies activation forgetting algorithms inplace given rate and its mode for given model
    Specificaly used for train_cl/train_pmr.py

    :param model: Current generator model (MLP)
    :param model_input: Inputs to the ``model`` (only used for ``gated_activations``)
    :param mode: Forgetting mode, either ``dropout`` or ``gated_activations``

    :return: None
    """

    # using model.eval for deactivating `dropout` will also deactivate `gated_activations`

    if mode == 'dropout':
        # TODO: it's not clean enough to add/remove dropouts dynamically here, so we just make sure dropouts are enabled
        # TODO: it is possible to easily change `rate` dynamically and it is fully readable. Extend this method to do that
        ## if using `dropout` helped at all.
        model.train()
    elif mode == 'gated_activations':
        model.register_gated_activations(model_input, rate=rate)
    else:
        raise ValueError('Activation forgetting "{}" does not exist! \n'.format(mode))

    sys.stderr.write('Activation forgetting "{} -> rate={}"  has been applied. \n'.format(mode, rate))