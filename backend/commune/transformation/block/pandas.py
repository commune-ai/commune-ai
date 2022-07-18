import numpy as np
import pandas as pd
from .base import BaseTransform

EPS = 1e-10

class rolling_ma(BaseTransform):
    def __init__(self, window=2, min_period=1):
        self.window = window
        self.min_period = min_period
    def transform(self, data):
        data = data.rolling(window=self.window, min_periods=self.min_period, center=True).mean()

        # fill nan with mean
        data = data.fillna(np.mean(data))

        return data


class temporal_difference(BaseTransform):
    def __init__(self, lag=1):
        self.lag = lag
    def transform(self, data):
        lag_data = data.shift(self.lag).fillna(0)
        return (data-lag_data)/(data+EPS)
