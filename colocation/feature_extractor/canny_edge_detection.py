"""Use canny edge detection to extract salient parts of the time series
"""
import logging
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import sobel

LOG = logging.getLogger("M.CannyEdge")


def detect_edge_scipy(data, sigma, lo=0.9, hi=0.99):
    """Apply canny edge detector from scipy image
    """
    from skimage import feature

    edge = np.empty_like(data)
    data = np.stack([data, data, data], axis=1)
    for row in range(data.shape[0]):
        _edge = feature.canny(
            data[row],
            sigma=sigma,
            low_threshold=lo,
            high_threshold=hi,
            use_quantiles=True,
        )
        edge[row] = _edge[1]
        del _edge
    return edge


def detect_edge(data, sigma, truncate, intermediate_out=None):
    """Apply gaussian filtering, sobel filtering, and threshold filtering
    """
    data = (
        _PipelinedArray(data, intermediate_out)
        .apply(
            lambda x: gaussian_filter1d(x, sigma, axis=1, truncate=truncate),
            label="Gaussian",
        )
        .apply(lambda x: sobel(x, axis=1), label="Sobel")
        # .apply(_non_maximum_suppression, label="Non-maximum Suppression")
        .apply(lambda x: _sigma_filter(x, 2), label="Filter by 2-sigma")
        .data
    )

    return data


def _sigma_filter(data, sigma):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    mask = np.abs(data - mean) > (sigma * std)
    return data * mask


def _non_maximum_suppression(array):
    larger_left = np.ones_like(array, dtype=np.bool)
    larger_right = np.ones_like(array, dtype=np.bool)
    less_left = np.ones_like(array, dtype=np.bool)
    less_right = np.ones_like(array, dtype=np.bool)

    larger_left[:, 1:] = array[:, 1:] > array[:, :-1]
    larger_right[:, :-1] = array[:, :-1] > array[:, 1:]
    less_left[:, 1:] = array[:, 1:] < array[:, :-1]
    less_right[:, :-1] = array[:, :-1] < array[:, 1:]

    maxima = larger_left & larger_right
    minima = less_left & less_right

    return array * (maxima | minima)


class _PipelinedArray(object):
    def __init__(self, original: np.ndarray, intermediate_out: dict = None):
        self.data = original
        self.intermediate_out = intermediate_out
        self._add_to_intermediate("orginial", original)

    def apply(self, func, label="Unamed"):
        """Applying a new function on the array
        """
        LOG.info("Pipelined Array applying funciton %s", label)
        self.data = func(self.data)
        self._add_to_intermediate(label, self.data)
        return self

    def _add_to_intermediate(self, label, data):
        if self.intermediate_out is not None:
            if label in self.intermediate_out:
                label = label + "+"
            self.intermediate_out[label] = np.copy(data)
