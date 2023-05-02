import os
import numpy as np
from pathlib import Path

"""Utility functions."""


def parse_path(path):
    """
    Parse the model saving path.
    :param path: the path to save trained models or estimate results.
    :return: two paths that save the best and last trained model/results.
    """
    model_best, model_last = '', ''
    if path != '':
        model_path = os.path.normpath(path).split(os.sep)
        model_name = model_path[-1]
        model_path[-1] = model_name + '_best'
        model_best = os.path.join(*model_path)
        model_path[-1] = model_name + '_last'
        model_last = os.path.join(*model_path)
    return model_best, model_last


def save_result(path, result, index):
    """Save the model estimate results."""
    Path(path).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(path, f'estimate{index + 1}.npz'), **result)
    return
