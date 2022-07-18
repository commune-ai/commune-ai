
"""
Encoders for encoding categorical variables and scaling continuous data.
"""
from typing import Iterable, Union
import warnings

from .base import BaseTransform
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch


class NaNLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Labelencoder that can optionally always encode nan and unknown classes (in transform) as class ``0``
    """

    def __init__(self, add_nan: bool = False, warn: bool = True):
        """
        init NaNLabelEncoder

        Args:
            add_nan: if to force encoding of nan at 0
            warn: if to warn if additional nans are added because items are unknown
        """
        self.add_nan = add_nan
        self.warn = warn
        self.fit_bool = False
        super().__init__()

    @staticmethod
    def is_numeric(y: pd.Series) -> bool:
        """
        Determine if sequence is numeric or not. Will also return True if sequence is a categorical type with
        underlying integers.

        Args:
            y (pd.Series): sequence for which to carry out assessment

        Returns:
            bool: True if sequence is numeric
        """
        return y.dtype.kind in "bcif" or (isinstance(y, pd.CategoricalDtype) and y.cat.categories.dtype.kind in "bcif")

    def fit(self, y: pd.Series):
        """
        Fit transformer

        Args:
            y (pd.Series): input data to fit on

        Returns:
            NaNLabelEncoder: self
        """

        self.classes_ = {}

        # determine new classes
        if self.add_nan:
            if self.is_numeric(y):
                nan = np.nan
            else:
                nan = "nan"
            self.classes_[nan] = 0
            idx = 1
        else:
            idx = 0

        for val in np.unique(y):
            if val not in self.classes_:
                self.classes_[val] = idx
                idx += 1

        self.classes_vector_ = np.array(list(self.classes_.keys()))
        self.fit_bool = True
        return self
        
    def transform(self, y: Iterable,) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode iterable with integers.

        Args:
            y (Iterable): iterable to encode
            return_norm: only exists for compatability with other encoders - returns a tuple if true.
            ignore_na (bool): if to ignore na values and map them to zeros
                (this is different to `add_nan=True` option which maps ONLY NAs to zeros
                while this options maps the first class and NAs to zeros)

        Returns:
            Union[torch.Tensor, np.ndarray]: returns encoded data as torch tensor or numpy array depending on input type
        """

        if not self.fit_bool:
            self.fit(y=y)

        if self.add_nan:
            if self.warn:
                cond = ~np.isin(y, self.classes_)
                if cond.any():
                    warnings.warn(
                        f"Found {np.unique(np.asarray(y)[cond]).size} unknown classes which were set to NaN",
                        UserWarning,
                    )

            encoded = [self.classes_.get(v, 0) for v in y]

        else:
            try:
                encoded = [self.classes_[v] for v in y]
            except KeyError as e:
                raise KeyError(
                    f"Unknown category '{e.args[0]}' encountered. Set `add_nan=True` to allow unknown categories"
                )

        if isinstance(y, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long, device=y.device)
        else:
            encoded = np.array(encoded)


        return encoded

    def inverse_transform(self, y: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Decode data, i.e. transform from integers to labels.

        Args:
            y (Union[torch.Tensor, np.ndarray]): encoded data

        Raises:
            KeyError: if unknown elements should be decoded

        Returns:
            np.ndarray: decoded data
        """
        if y.max() >= len(self.classes_vector_):
            raise KeyError("New unknown values detected")

        # decode
        decoded = self.classes_vector_[y]
        return decoded


from sklearn.preprocessing import KBinsDiscretizer
import streamlit as st


def equal_intervals_pandas_series(series, nbins=10):
    max = series.max()
    min = series.min()

    bin_size = (max - min) / nbins

    for bin_id in range(nbins):
        bin_bounds = [min + bin_id * bin_size,
                      min + (bin_id + 1) * bin_size]
        series = series.apply(lambda x: bin_bounds[0] if x >= bin_bounds[0] and x < bin_bounds[1] else x)

    return series



class NumericDiscretizer(BaseTransform):
    fit_bool = False
    def __init__(self,
                 nbins=5,
                 encoder='ordinal',
                 strategy='uniform'):
        self.nbins = nbins
        self.encoder = encoder
        self.strategy = strategy
        self.est = KBinsDiscretizer(n_bins=nbins, encode=encoder, strategy=strategy)

    def fit(self, data):

        n_unique_elements = len(np.unique(data))
        if  n_unique_elements < self.nbins:
            self.est = KBinsDiscretizer(n_bins=int(n_unique_elements * 0.95),
                                        encode=self.encoder,
                                        strategy=self.strategy)
        self.est.fit(data)
        self.fit_bool = True

    def transform(self, data):
        original_data_shape = data.shape
        if len(original_data_shape) == 1:
            data = data[:, None]
        if not self.fit_bool:
            self.fit(data)
        data = self.est.transform(data)

        if len(original_data_shape) == 1:
            data = data[:,0]


        data = self.est.__dict__['bin_edges_'][0][data.astype(int)]


        return data

    def inverse(self, data):
        assert self.fit_bool
        self.est.inverse_transform(data)
        return data


