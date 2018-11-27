"""Process, reshape raw data from sensors
"""

from typing import List

import numpy as np


def trim_and_align(data_list: List[np.ndarray], interval: int) -> np.ndarray:
    """Trim a list of data, so they start and end at the same time

    Args:
        data_list (List[np.ndarray]): a list of data. Each
            data item should be of the shape [ ? x 2 ].
            The first column contains time stamps and the second column contains values.
        interval (int): the interval of time stamp.
            If the value is None, then interval will equal the mean of all intervals in
            the first series. If the mean interval is not int, then it will be floored.

    Returns:
        np.ndarray: [m x ?]
    """
    assert data_list, "There must be at least one series"
    assert (
        data_list[0].ndim == 2
    ), "Expect 2-dimension series. Instead, ndim = {}".format(
        data_list[0].ndim
    )

    max_start_time = max(d[0, 0] for d in data_list)
    min_end_time = min(d[-1, 0] for d in data_list)

    new_axis = np.arange(max_start_time, min_end_time + 1, int(interval), dtype=int)

    new_series = np.empty((len(data_list), len(new_axis)), dtype=float)

    for i, data in enumerate(data_list):
        new_series[i] = np.interp(new_axis, data[:, 0], data[:, 1])

    return new_series


def autocorr(array):
    """Calculate autocorrelation of a 1D array
    """
    assert array.ndim == 1, "array must be 1D"
    corr = np.correlate(array, array, mode="full")
    return corr[corr.size // 2 :]


def normalize(dataset):
    """Normalize a dataset
    """
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    return (dataset - mean) / std


def scale(dataset, new_max, new_min):
    """Scale and translate a dataset uniformly to fit in a maxmin constraint
    """
    d_max = np.max(dataset, axis=1)
    d_min = np.min(dataset, axis=1)
    dataset = (dataset - d_min) / (d_max - d_min) * (new_max - new_min) + new_min
    return dataset


def suppress_noise(dataset, std_threshold):
    """suppress noisy background by zero-ing fluctuations within a threshold
    """
    ret_val = np.zeros_like(dataset)
    means = np.mean(dataset, axis=1)
    one_std = np.std(dataset, axis=1)

    for row in range(dataset.shape[0]):
        copy_ids = np.where(np.abs(dataset[row] - means[row]) > one_std[row])
        ret_val[row][copy_ids] = dataset[row][copy_ids]

    return ret_val
