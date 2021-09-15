import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import matplotlib as mpl
mpl.use('Agg')


def prepare_task_values(interval=5, start=0, end=10, order='ctf'):
    """
    Prepare an array of changes in task_values given interval and range. 
    Resoluions to train with - all of them follow same ratio using ``domainCorners`` in ``top``.
    Scales do not have any constraints

    :param interval: Interval between each sampled resolution
    :param start: Smallest resolution
    :param end: Largest resolution
    :param order: 1. 'ctf': coarse-to-fine (increasing order)
                  2. 'ftc': fine-to-coarse (decreasing order)
                  3. 'bidirectional': Zigzag between coarse and fine
                  4. 'random': A random order of task_values
                  5. 'manual: An array of indices providing the order [WIP]
    
    """

    task_values = np.arange(start=start, stop=end) * interval

    if order == 'ctf':
        return task_values
    elif order == 'ftc':
        return -task_values
    elif order == 'bidirectional':
        raise NotImplementedError('Not yet!')
    elif order == 'random':
        np.random.shuffle(task_values)
        return task_values
    elif order == 'manual':
        raise NotImplementedError('Not yet!')
    else:
        raise NotImplementedError('Mode does not exist or has not been implemented yet!')

