from time import time

import numpy as np
import torch as th

# from torch.autograd.functional import jacobian


def timer(func):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f"\n{func.__name__} cost time {time_spend} s\n")
        return result

    return func_wrapper


class DataNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0, ddof=1)

    def norm(self, data, index=None):
        assert (self.mean is not None) and (self.std is not None)
        if index is None:
            return (data - self.mean) / self.std
        else:
            assert data.shape[0] == index.shape[0]
            return (data - self.mean[index]) / self.std[index]

    def denorm(self, data, index=None):
        assert (self.mean is not None) and (self.std is not None)
        if index is None:
            return data * self.std + self.mean
        else:
            assert data.shape[0] == index.shape[0]
            return data * self.std[index] + self.mean[index]

    @staticmethod
    def fit_norm(data):
        std = np.nanstd(data, axis=0, ddof=1)
        mean = np.nanmean(data, axis=0)
        return (data - mean) / std


def remove_finite(x: th.Tensor):
    return th.where(
        ~(x.isfinite()),
        th.zeros_like(x),
        x)


def num_curl_loss(jcb_mtx):
    s_error = remove_finite(jcb_mtx - jcb_mtx.permute(0, 1, 2, 4, 3)).square()
    curl_loss = s_error.sum(dim=-1).sum(dim=-1).median()
    return curl_loss
