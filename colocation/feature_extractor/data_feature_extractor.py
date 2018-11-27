"""From data_helper
"""
import numpy as np
import scipy as sp

from scipy import stats
from collections import Counter, defaultdict
from multiprocessing import Pool

"""for Calbimonte"""


def get_SS(X, B=2):
    """Turn raw sequences to sequences of segments
    
    Args:
        X: A list of sequences. 
        B (int, optional): Defaults to 2. 
    
    Returns:
        List[List[Tuple[int, int, float, float, int, int]]]: A list of sequences of
            segments. Each segments is composed of:
                starting id, ending id,
                maximum value, minimum value,
                id of maximum value, id of minimum value
    """

    pool = Pool()

    N, D = X.shape

    # Process X line by line
    SS = pool.map(getS_wrapper, [(X[i, :], B) for i in range(N)])

    pool.close()
    pool.join()

    return SS


def getS_wrapper(args):
    """Tuple wrap the arguments, because mp.Pool can only one argument
    """
    return getS(*args)


def getS(ts, B):
    """Get a sequence of $B * 2$ buckets. 

    If ts is longer than $(B*2)$, neighboring buckets with lowest merging error
    will be merged.

    Each bucket is a segment of time series, with a max and min attached.
    """
    # ts: a line of time series
    # S: a series of buckets. Maintains the length ``B*2``.
    S = [(i, i, ts[i], ts[i], i, i) for i in range(0, 2 * B)]
    for i in range(2 * B, len(ts)):
        S.append((i, i, ts[i], ts[i], i, i))
        S = merge_neighbour_buckets(S)
    return S


def merge_neighbour_buckets(S):
    """Merge two neighboring buckets in the sequence, who have the lowest merged error.

    S will be modified, with the two merged buckets replaced by the result of merging.
    """
    # 2B+1 buckets now, calculate the total error of merging neighbour buckets
    # ? What is error?
    Err = []  # error of merging different buckets

    # Get maximum error of unmerged buckets
    errs = [get_bucket_err(this_bucket) for this_bucket in S]  # errors of each bucket
    max_err = max(errs)

    # Get errors of merged buckets
    for i in range(len(S) - 1):
        temp_max_err = max_err

        # Error of merged bucket
        temp_err = get_bucket_err(merge_two_buckets(S[i], S[i + 1]))

        # Append this error only if it is larger than max error of unmerged bucket
        temp_max_err = temp_err if temp_err > temp_max_err else max_err
        Err.append(temp_max_err)

    # Find the minimum error in merged buckets
    merge_idx = Err.index(min(Err))

    # Replace merged buckets with the resulting bucket
    S[merge_idx] = merge_two_buckets(S[merge_idx], S[merge_idx + 1])
    S.pop(merge_idx + 1)
    return S


def get_bucket_err(s):
    # ? is this Max - min?
    return (s[2] - s[3]) / 2


def merge_two_buckets(s1, s2):
    """Merge two buckets

    1. Extending the range: from ``min(beg(s1), beg(s2))`` to ``max(end(s1), end(s2))``
    2. Resetting max and min, and max_id and min_id

    Args:
        s1: sequence 1
        s2: sequence 2

    Returns:
        tuple: a merged bucket
    """

    s1max = max(s1[2], s2[2]) == s1[2]
    s1min = min(s1[3], s2[3]) == s1[3]
    # get max
    max_val = s1[2] if s1max else s2[2]
    max_val_index = s1[4] if s1max else s2[4]
    # get min
    min_val = s1[3] if s1min else s2[3]
    min_val_index = s1[5] if s1min else s2[5]

    # ? The first two element: begin to end?
    return (
        min(s1[0], s2[0]),
        max(s1[1], s2[1]),
        max_val,
        min_val,
        max_val_index,
        min_val_index,
    )


def get_piecewise_linear_symbol_feature(slopes, segs=4):
    """Digitize gradients

    Given slopes, first group them into bins.
    Next, count the number for each bin.

    Return: array of int, each corresponding to the number of element in each bin.
    """

    bins = np.linspace(-np.pi / 2, np.pi / 2, segs + 1)
    symbols = np.digitize(slopes, bins)
    c = Counter(symbols)
    return np.array([c[i + 1] for i in range(segs)])


def get_ts_slopes(S):
    """Collect slopes along a time series
    
    Args:
        S: a list of buckets representing a time series
    
    Returns:
        slopes: an array of slopes within each bucket
    """

    return np.array([get_bucket_slope(s) for s in S])


def get_bucket_slope(a):
    """Calcualte the gradient in a bucket
    
    A bucket has: start_id, end_id, max, min, max_id, min_id

    Slope = | 0                                       if start_id = end_id
            | (max - min) / (end_id - start_id)       if max_id >= min_id
            | - (max - min) / (end_id - start_id)     if min_id > max_id

    Args:
        a (tuple): a bucket
    
    Returns:
        float : Gradient
    """

    return (
        np.arctan((a[2] - a[3]) / (a[1] - a[0]) * np.sign(a[4] - a[5]))
        if a[0] != a[1]
        else 0
    )


"""for Gao"""


def mode(ndarray, axis=0):
    """[summary]
    TODO: have to read more on Gao's ``mode()``
    Args:
        ndarray ([type]): [description]
        axis (int, optional): Defaults to 0. [description]
    
    Raises:
        Exception: If the array is empty, or if axis is larger or equal to dimension.

    Returns:
        [type]: [description]
    """

    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception("Attempted to find mode on an empty array!")

    # If axis >= dimension, raise Exception
    try:
        axis = [i for i in range(ndarray.ndim)][axis]
    except IndexError:
        raise Exception(
            "Axis %i out of range for array with %i dimension(s)" % (axis, ndarray.ndim)
        )

    srt = np.sort(ndarray, axis=axis)
    dif = np.diff(srt, axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1, -1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices, axis=axis)
    location = np.argmax(bins, axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [
        mesh[i][index].ravel() if i != axis else location.ravel()
        for i in range(bins.ndim)
    ]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)

    return (modals, counts)


"""for Hong"""


def get_statF_on_window(X):
    """Get some stats on a window in several time series
    
    Args:
        X: a window of time series
    
    Returns:
        array: 11 statistics:
        [ 0]: min
        [ 1]: median
        [ 2]: sqrt(mean(square))
        [ 3]: max
        [ 4]: variance
        [ 5]: skew
        [ 6]: kurtosis
        [ 7]: slope
        [ 8]: 25% percentile
        [ 9]: 75% percentile
        [10]: 75% percentile - 25% percentile
    """

    N, D = X.shape
    dim = 11
    F = np.zeros([N, dim])
    # percentiles to be used
    p = [25, 75]

    F[:, 0] = np.min(X, 1)
    F[:, 1] = np.median(X, 1)
    F[:, 2] = np.sqrt(np.mean(np.square(X), 1))  # ? Why not along 1 for mean and sqrt
    F[:, 3] = np.max(X, 1)
    F[:, 4] = np.var(X, 1)
    F[:, 5] = sp.stats.skew(X, 1)
    F[:, 6] = sp.stats.kurtosis(X, 1)

    # calculate slope
    xx = np.linspace(1, D, D)
    tempx = xx - np.mean(xx)
    F[:, 7] = tempx.dot((X - np.mean(X)).T) / (tempx.dot(tempx.T))

    # quantiles
    F[:, 8 : len(p) + 8] = np.vstack([np.percentile(X, i, axis=1) for i in p]).T
    F[:, 10] = F[:, 9] - F[:, 8]

    # check illegal features nan/inf
    F[np.isnan(F)] = 0
    F[np.isinf(F)] = 0

    return F


def window_feature(X, feature_fun, win_num, overlapping=0):
    """function used to extract features by window sections and concatenate them
    """
    if win_num < overlapping:
        print("Error! overlapping length should be smaller than window length")
    N, D = X.shape
    temp = feature_fun(X[:2, :10])
    _, dimf = temp.shape
    F = np.zeros([N, dimf, D // (win_num - overlapping)])
    cnt = 0
    for i in range(0, D - 1, win_num - overlapping):
        start = i if i < overlapping else i - overlapping
        temp = feature_fun(X[:, start : start + win_num])
        F[:, :, cnt] = temp
        cnt = cnt + 1
    return F


"""for Balaji"""


def haar_transform(x):
    xc = x.copy()
    n = len(xc)

    avg = [0 for i in range(n)]
    dif = [0 for i in range(n)]

    while n > 1:

        for i in range(int(n / 2)):
            avg[i] = (xc[2 * i] + xc[2 * i + 1]) / 2
            dif[i] = xc[2 * i] - avg[i]

        for i in range(int(n / 2)):
            xc[i] = avg[i]
            xc[i + int(n / 2)] = dif[i]

        n = int(n / 2)

    return xc


class data_feature_extractor:
    def __init__(self, X):
        self.X = X
        self.functions = [
            "getF_1994_Li",
            "getF_2012_Calbimonte",
            "getF_2015_Gao",
            "getF_2015_Hong",
            "getF_2015_Bhattacharya",
            "getF_2015_Balaji",
            "getF_2016_Koh",
        ]

    # Feature
    def getF_1994_Li(self):
        """ 'mean','variance','CV' (coefficient of variation) """
        X = self.X
        N, D = X.shape
        dim = 3
        F = np.zeros([N, dim])

        F[:, 0] = np.mean(X, 1)
        F[:, 1] = np.var(X, 1)
        F[:, 2] = np.std(X, 1) / np.mean(X, 1)

        names = ["mean", "variance", "CV"]

        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F

    # Feature
    def getF_2012_Calbimonte(self, B=20, segs=5):
        X = self.X

        SS = get_SS(X, B)

        PLSF = np.array(
            [get_piecewise_linear_symbol_feature(get_ts_slopes(S), segs) for S in SS]
        )
        PLSF = PLSF.astype(float)

        return PLSF

    # Feature
    def getF_2015_Gao(self):
        """ 'min','median','mean','max','std','skewness','kurtosis','entropy','percentile'. """
        X = self.X

        N, D = X.shape
        dim = 15
        F = np.zeros([N, dim])
        # percentiles to be used
        p = [2, 9, 25, 75, 91, 98]

        F[:, 0] = np.min(X, 1)
        F[:, 1] = np.median(X, 1)
        F[:, 2] = np.mean(X, 1)
        F[:, 3] = np.max(X, 1)
        F[:, 4] = np.std(X, 1)
        F[:, 5] = sp.stats.skew(X, 1)
        F[:, 6] = sp.stats.kurtosis(X, 1)

        # digitize the data for the calculation of entropy if it only contains less than 100 discreate values
        XX = np.zeros(X.shape)
        bins = 100
        for i in range(X.shape[0]):
            if len(np.unique(X[i, :])) < bins:
                XX[i, :] = X[i, :]
            else:
                XX[i, :] = np.digitize(
                    X[i, :], np.linspace(min(X[i, :]), max(X[i, :]), num=bins)
                )
        F[:, 7] = sp.stats.entropy(XX.T)

        F[:, 8 : len(p) + 8] = np.vstack([np.percentile(X, i, axis=1) for i in p]).T

        F[:, 14] = mode(X, 1)[0]

        names = [
            "min",
            "median",
            "mean",
            "max",
            "std",
            "skewness",
            "kurtosis",
            "entropy",
            "p2",
            "p9",
            "p25",
            "p75",
            "p91",
            "p98",
            "mode",
        ]

        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F

    # Feature
    def getF_2015_Hong(self):
        X = self.X
        F = window_feature(X, get_statF_on_window, 4, overlapping=2)
        return np.hstack([np.min(F, 2), np.max(F, 2), np.median(F, 2), np.var(F, 2)])

    # Feature
    def getF_2015_Bhattacharya(self):
        X = self.X
        mean_var_fun = lambda x: np.vstack([np.mean(x, 1), np.var(x, 1)]).T
        F = window_feature(X, mean_var_fun, 3, overlapping=0)
        return np.hstack([np.min(F, 2), np.max(F, 2), np.median(F, 2), np.var(F, 2)])

    # Feature
    def getF_2015_Balaji(self):
        X = self.X
        N, D = X.shape
        dim = 24
        F = np.zeros([N, dim])

        # 1)scale based: mean/max/min/quartiles/range;
        F[:, 0] = np.mean(X, 1)
        F[:, 1] = np.max(X, 1)
        F[:, 2] = np.min(X, 1)
        F[:, 3] = np.percentile(X, 25, axis=1)
        F[:, 4] = np.percentile(X, 75, axis=1)
        F[:, 5] = F[:, 1] - F[:, 2]

        # 2)pattern based: 3 Haar wavelets and 3 Fourier coefficients;
        F[:, 6:9] = haar_transform(X)[:, :3]  # this does not seem to be right
        F[:, 9:12] = abs(np.fft.fft(X, axis=1)[:, 1:4]) / D  # 0-th is the average

        # 3)shape based: location and magnitude of top 2 components from piece-wise constant model, error variance;
        F[:, 12:18] = haar_transform(X)[:, 4:10]

        # 4)texture based: first and second var of difference between consecutive samples, max var,
        # number of up and down changes, edge entropy measure
        F[:, 18] = np.var(np.diff(X, n=1, axis=1), 1)  # first difference
        F[:, 19] = np.var(np.diff(X, n=2, axis=1), 1)  # second difference
        # max variation??
        F[:, 20] = np.var(X, 1)
        # number of ups
        ct = Counter(np.where(np.diff(X, n=1, axis=1) > 0)[0])
        F[:, 21] = [ct[i] for i in range(N)]
        # number of downs
        ct = Counter(np.where(np.diff(X, n=1, axis=1) < 0)[0])
        F[:, 22] = [ct[i] for i in range(N)]
        # edge entropy
        # digitize the data for the calculation of entropy if it only contains less than 100 discreate values
        XX = np.zeros(X.shape)
        bins = 100
        for i in range(N):
            if len(np.unique(X[i, :])) < bins:
                XX[i, :] = X[i, :]
            else:
                XX[i, :] = np.digitize(
                    X[i, :], np.linspace(min(X[i, :]), max(X[i, :]), num=bins)
                )
        F[:, 23] = sp.stats.entropy(XX.T)

        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F

    # Feature
    def getF_2016_Koh(self):
        X = self.X
        """ 'mean','var','dominant freq','skewness','kurtosis' """
        N, D = X.shape
        dim = 7
        F = np.zeros([N, dim])

        F[:, 0] = np.mean(X, 1)
        F[:, 1] = np.var(X, 1)
        F[:, 2] = np.mean(X, 1)

        temp_fft = abs(np.fft.fft(X, axis=1)) / D
        F[:, 3:5] = np.vstack([temp_fft[i, :].argsort()[-3:-1][::-1] for i in range(N)])

        F[:, 5] = sp.stats.skew(X, 1)
        F[:, 6] = sp.stats.kurtosis(X, 1)

        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F
