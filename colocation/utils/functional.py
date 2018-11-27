"""Some function helpers
"""
from typing import Iterable
import numpy as np


def noop(*_, **__):
    """Do nothing
    """
    pass


def array_map(func, iterable: Iterable, out_shape, dtype=None, update_callback=noop):
    """Map a function on to an iterable of numpy array
    """
    assert isinstance(iterable, (list, np.ndarray))
    dtype = dtype if dtype else iterable[0].dtype
    result = np.zeros(out_shape, dtype=dtype)

    for i, entry in enumerate(iterable):
        result[i] = func(entry)
        update_callback(i)

    return result


def window_count(total_length, window_size, stride):
    """Calculate the number of windows
    """
    return round((total_length - window_size + 1) / stride)
