import numpy as np
import pandas as pd
import torch
import pywt
from .base import BaseTransform
EPS = 1e-10

class minmax_scalar(BaseTransform):
    fit_bool = False

    def __init__(self,
                 min = None,
                 max= None,
                 min_val=-1,
                 max_val=1):
        self.min=min
        self.max = max

        self.min_val = min_val
        self.max_val = max_val

        assert max_val > min_val

    def fit(self, data):

        if self.min is None:
            self.min = np.min(data)
        if self.max is None:
            self.max = np.max(data)


        self.fit_bool = True

    def transform(self, data):

        if not self.fit_bool:
            self.fit(data)
        data = ((data-self.min)/(self.max-self.min))

        data =  self.min_val + data*(self.max_val - self.min_val)

        return data

    def inverse(self,data):
        assert self.fit_bool

        # uncenter the min max value if it is not cented
        data = (data - self.min_val)/(self.max_val - self.min_val)

        return data

class standard_variance(BaseTransform):
    fit_bool = False
    def __init__(self):
        self.std = None
    def fit(self, data):
        self.std= np.std(data)
        self.fit_bool = True

    def transform(self, data):
        if not self.fit_bool:
            self.fit(data)

        data = data/self.std
        return data


    def inverse(self, data):
        assert self.fit_bool
        data = data*self.std
        return data

class standard_scalar(BaseTransform):
    fit_bool = False

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):

        self.mean = np.mean(data)
        self.std = np.std(data)

        self.fit_bool = True

    def transform(self, data):

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if not self.fit_bool:
            self.fit(data)

        data = (data - self.mean) / self.std
        return data

    def inverse(self, data):
        assert self.fit_bool
        data = (data * self.std) + self.mean
        return data

class log_transform(BaseTransform):
    fit_bool = False

    def __init__(self):
        self.min = None

    def transform(self, data):
        # apply log pipeline
        data = np.log(data)

        return data

    def inverse(self, data):
        assert self.fit_bool
        data = np.exp(data)
        return data

class temporal_difference(BaseTransform):
    def __init__(self, lag=1):
        self.lag = lag

    def transform(self, data):
        if isinstance(data, list):
            data = np.array(data)

        back_shift = data[:-self.lag]
        forward_shift = data[self.lag:]
        return (forward_shift-back_shift)

class temporal_percent_difference(BaseTransform):
    def __init__(self, lag=1):
        self.lag = lag

    def transform(self, data):
        if isinstance(data, list):
            data = np.array(data)

        back_shift = data[:-self.lag]
        forward_shift = data[self.lag:]
        return (forward_shift-back_shift)/(forward_shift+EPS)

class wavelet_filter(BaseTransform):
    def __init__(self, thresh=0.63, wavelet="db4"):
        self.thresh = thresh
        self.wavelet = wavelet
    def transform(self,x):
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        thresh = self.thresh * np.nanmax(x)
        coeff = pywt.wavedec(x, self.wavelet, mode="per")
        coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
        reconstructed_x = pywt.waverec(coeff, self.wavelet, mode="per")
        return reconstructed_x
