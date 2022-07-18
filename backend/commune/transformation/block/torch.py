"""

PYTORCH FUNCTIONS

"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from scipy.signal import savgol_filter
from .base import BaseTransform, input_schema_wrapper
import pywt
from contextlib import contextmanager
import streamlit as st
from commune.model.block.gp.block import Sequence_GP_Extrapolator

EPS = 10E-10

class low_pass_fft(BaseTransform):
    def __init__(self,
                 lowest_freq_frac=0.5,
                 buffer_period_frac=0.2,
                 batch=True):

        self.lowest_freq_frac=lowest_freq_frac
        self.buffer_period_frac=buffer_period_frac
        self.batch = batch

    @input_schema_wrapper(torch.Tensor)
    def transform(self, sig):
        from scipy import fftpack
        # The FFT of the signal

        if len(sig.shape) == 1:
            sig = sig.unsqueeze(0)

        original_length = sig.shape[1]

        buffer_period = int(self.buffer_period_frac*original_length)

        sig = torch.cat([*[sig[:, :1]] * buffer_period, sig, *[sig[:, -1:]] * buffer_period], dim=1)

        sig_fft = torch.fft.rfft(sig, dim=1)

        lowest_freqs = int(self.lowest_freq_frac*sig_fft.shape[1])

        low_freq = torch.cat([sig_fft[:, :lowest_freqs], sig_fft[:, lowest_freqs:] * 0], dim=1)
        filtered_sig = torch.fft.irfft(low_freq, dim=1).float()

        out_tensor = filtered_sig[:, buffer_period:buffer_period + original_length]

        if not self.batch and len(out_tensor.shape)==2:
            out_tensor = out_tensor.squeeze(0)

        return out_tensor


class difference_transform(BaseTransform):
    def __init__(self,
                 step=1,
                 padding='left',
                 batch_first=True,
                 relative_difference=True):

        self.__dict__.update(locals())

    @input_schema_wrapper(torch.Tensor)
    def transform(self, x):

        """temporal difference on a numpy, tensor function"""

        if len(x.shape) == 1:
            self.batch_first = False
            x = x.unsqueeze(0)

        # padding to avoid reducing sequenc  dimension


        if self.padding == 'left':
            padding = x[:, :1].repeat(1, self.step)
            x = torch.cat([padding,x], dim=1)
        if self.padding == 'right':
            padding = x[:, -1:].repeat(1, self.step)
            x = torch.cat([x,padding], dim=1)

        lagged_x = x[..., :-self.step]
        forward_x = x[..., self.step:]
        x_diff = forward_x - lagged_x

        if self.relative_difference:
            x_diff = x_diff / (lagged_x + EPS)

        if not self.batch_first:
            x_diff = x_diff.squeeze(0)

        return x_diff

class gp_extrapolate(BaseTransform):
    def __init__(self,
                 batch_size=2048,
                 lr=0.01,
                 sample_freq=1.0,
                 steps = 10,
                 extension_coeff=[0.2, 0.2],
                 device = 'cuda',
                 verbose=False,
                 batch=False):

        '''

        :param batch_size:
            - batch size for batched regression tools
        :param lr:
            - learning rate of gp
        :param sample_freq:
            - sample frequency for training gp
        :param extension_coeff:
            - extension coefficients in extrapolation for sequence ([left, right])
        :param device:
        '''

        self.batch_size = batch_size
        self.lr = lr
        self.sample_freq = sample_freq
        self.extension_coeff = extension_coeff
        self.steps = steps
        self.verbose = verbose
        self.batch = batch

        # ensure gpu is available if using cuda (TEMP FIX)

        if "cuda" in device:
            if not torch.cuda.is_available():
                device = 'cpu'

        self.device = device

    @input_schema_wrapper(torch.Tensor)
    def transform(self, x):
        """
        extrapolates x

        :param x: (batch of x, N) tensor (numpy or torch.tensor)
        :return: (batch of x , N * (sum(extension_coeff) + 1) ) (numpy or torch.tensor)

        """

        x = x.float()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        assert len(x.shape) == 2, "dimension should be 2 (batchx(sequence length))"

        x_batch_size, sequence_length = x.shape


        batch_size = min(x_batch_size, self.batch_size)


        # batch pipeline the input (if length of x is larger than batch size)
        output_batch_list = []

        for batch_start_idx in tqdm(range(0, x_batch_size, batch_size)):

            batch_end_idx = batch_start_idx + batch_size
            x_batch = x[batch_start_idx:batch_end_idx]

            # if the batch is truncated we need to feed the correct batch into the gp
            # this is because a parallel gp is being fit onto each batch

            gp_batch_size = x_batch.shape[0]
            # TODO: refresh parameters for each batch instead of recompiling the model
            gp = Sequence_GP_Extrapolator(batch_size=gp_batch_size,
                                          lr = self.lr,
                                          steps=self.steps).to(self.device)


            # extend the batch

            extension_periods = list(map(lambda x: int(x*sequence_length),
                                    self.extension_coeff))

            self.meta["extension_periods"] = extension_periods

            extended_batch = gp(y=x_batch.to(self.device),
                                 sample_freq = self.sample_freq,
                                 extension_periods = extension_periods,
                                 )



            output_batch_list.append(extended_batch)

        # concat the batches
        output_batch = torch.cat(output_batch_list, dim=0 )

        if not self.batch:
            output_batch = output_batch.squeeze(0)


        return output_batch

import streamlit as st
class truncate_extrapolation(BaseTransform):
    def __init__(self, extension_periods=None,
                 batch_first= True):
        self.extension_periods = extension_periods
        self.batch_first = batch_first
    @input_schema_wrapper(torch.Tensor)
    def transform(self, data):
        if self.extension_periods:
            extension_periods = self.extension_periods

        else:
            assert 'extension_periods' in self.meta, \
                'BRUHHHHH, specify the extension_periods in ' \
                'either self.meta or the init fam'

            extension_periods = self.meta['extension_periods']

        if self.batch_first:
            data = data[extension_periods[0]:data.shape[0]  - extension_periods[-1]]
            #data = data[extension_periods[0]:-extension_periods[-1]]
        else:
            data = data[:, extension_periods[0]:data.shape[1]-extension_periods[-1]]
        return data



class step_indexer(BaseTransform):
    def __init__(self, step, dim=1, avg_pool=True):
        self.step = step
        self.dim = dim
        self.avg_pool =avg_pool

    @input_schema_wrapper(torch.Tensor)
    def transform(self, x):
        assert len(x.shape) == 2, "Must be Batch Size x Sequence Length"

        batch_size, seq_len = x.shape



        if self.avg_pool:
            x = torch.nn.AvgPool1d(kernel_size=self.step,
                                   stride=self.step)(x.unsqueeze(1)).squeeze(1)
        else:
            step_index = torch.arange(0, seq_len, self.step)
            x = torch.index_select(x, self.dim, step_index)

        return x


class savgol_filter_transform(BaseTransform):
    def __init__(self,window_length,
                 polyorder,
                 deriv=0,
                 delta=1.0,
                 axis=- 1,
                 mode='interp',
                 cval=0.0):

        self.params = dict(
            window_length=window_length + int((window_length % 2) == 0) ,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            axis= axis,
            mode=mode,
            cval=cval
        )

    @input_schema_wrapper(np.ndarray)
    def transform(self, x):
        """temporal difference on a numpy, tensor function"""

        x = savgol_filter(x, **self.params)

        return x

class wavelet_filter(BaseTransform):
    def __init__(self, thresh=0.63, wavelet="db4"):
        self.thresh = thresh
        self.wavelet = wavelet

    @input_schema_wrapper(np.ndarray)
    def transform(self,x):

        thresh = self.thresh * np.nanmax(x)
        coeff = pywt.wavedec(x, self.wavelet, mode="per")
        coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
        reconstructed_x = pywt.waverec(coeff, self.wavelet, mode="per")

        return reconstructed_x


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

    @input_schema_wrapper(torch.Tensor)
    def fit(self, data):

        if self.min is None:
            self.min = torch.min(data)
        if self.max is None:
            self.max = torch.max(data)


        self.fit_bool = True

    @input_schema_wrapper(torch.Tensor)
    def transform(self, data):

        if not self.fit_bool:
            self.fit(data)
        data = ((data-self.min)/(self.max-self.min))

        data =  self.min_val + data*(self.max_val - self.min_val)

        return data

    @input_schema_wrapper(torch.Tensor)
    def inverse(self,data):
        assert self.fit_bool

        # uncenter the min max value if it is not cented
        data = (data - self.min_val)/(self.max_val - self.min_val)

        return data

class standard_variance(BaseTransform):
    fit_bool = False
    def __init__(self):
        self.std = None

    @input_schema_wrapper(torch.Tensor)
    def fit(self, data):
        self.std= torch.std(data)

        self.fit_bool = True

    @input_schema_wrapper(torch.Tensor)
    def transform(self, data):
        if not self.fit_bool:
            self.fit(data)

        # print("BEFORE VARIANCE TRANSFORM: ",data.std(), data.mean(),  self.std   )
        data = data/self.std

        # print("AFTER VARIANCE TRANSFORM: ",data.std(), data.mean(), self.std )

        return data

    @input_schema_wrapper(torch.Tensor)
    def inverse(self, data):
        assert self.fit_bool
        data = data*self.std
        return data

class standard_scalar(BaseTransform):
    fit_bool = False

    def __init__(self):
        self.mean = None
        self.std = None

    @input_schema_wrapper(torch.Tensor)
    def fit(self, data):

        self.mean = torch.mean(data)
        self.std = torch.std(data)

        self.fit_bool = True

    @input_schema_wrapper(torch.Tensor)
    def transform(self, data):

        if not self.fit_bool:
            self.fit(data)

        data = (data - self.mean) / self.std
        return data

    @input_schema_wrapper(torch.Tensor)
    def inverse(self, data):
        assert self.fit_bool
        data = (data * self.std) + self.mean
        return data

class log_transform(BaseTransform):
    fit_bool = False

    def __init__(self):
        self.min = None

    @input_schema_wrapper(torch.Tensor)
    def transform(self, data):
        # apply log pipeline
        data = torch.log(data)

        return data

    @input_schema_wrapper(torch.Tensor)
    def inverse(self, data):
        assert self.fit_bool
        data = torch.exp(data)
        return data

