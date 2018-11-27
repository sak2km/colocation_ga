"""Build some wrappers around progressbar2
"""


import progressbar
from . import functional


def _set_up_bar(max_value, initial_value):
    pbar = progressbar.ProgressBar(max_value=max_value)
    pbar.update(initial_value)
    return pbar


def get_update_function(max_value, initial_value=0, noop=True):
    """Get a update function that prints progress
    """
    return functional.noop if noop else _set_up_bar(max_value, initial_value).update
