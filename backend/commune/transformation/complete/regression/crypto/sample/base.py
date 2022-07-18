"""
Sample Wise Transformer
"""

import torch
from commune.transformation.block.base import SequentialTransformPipeline
from commune.transformation.block.numpy import standard_scalar, minmax_scalar
from commune.transformation.block.torch import (savgol_filter_transform,
                                              low_pass_fft,
                                              gp_extrapolate,
                                              difference_transform,
                                              truncate_extrapolation,
                                              step_indexer)
from copy import deepcopy
import pandas as pd
import itertools


import os
import numpy as np

class SampleTransformManager(object):
    def __init__(self,
                 feature_group,
                 gt_keys,
                 meta_keys,
                 periods,
                 known_future_features,
                 process_pipeline_map=None):


        self.__dict__.update(locals())



        self.index_bounds = [
            0,
            (self.periods["input"]) * self.periods['step'],
            (self.periods["input"] + self.periods['output']) * self.periods['step']

        ]
        if not self.process_pipeline_map:
            self.build()
    def build(self):
        """
        Build  mapping for tensor key to transform pipeline
        """

        process_pipeline_map = {}
        # price sample wise preprocessing for prices
        for key in self.feature_group['con']:
            process_pipeline_map[key] =  \
                {'pipeline':  SequentialTransformPipeline(pipeline=[
                        savgol_filter_transform(window_length=17,
                                                polyorder=3),
                        low_pass_fft(lowest_freq_frac=0.2,
                                     buffer_period_frac=0.2)],),
                'input_key': key}
        self.process_pipeline_map = process_pipeline_map

    def transform(self, x):

        """

        :param x: dictionary of tensors
        :return: dictionary of tensors post transformed
        """

        for key in x.keys():
            if key in self.process_pipeline_map:
                pipeline_state_dict = self.process_pipeline_map[key]
                input_key = pipeline_state_dict['input_key']
                pipeline_obj = pipeline_state_dict['pipeline']
                x[key] = pipeline_obj.transform(deepcopy(x[input_key]))
        return x

    def get_inputs(self, x):
        x = deepcopy(x)
        out_dict = {}

        past_out_dict = {}
        future_out_dict = {}

        for feature_list in self.feature_group.values():
            for feature in feature_list:
                if feature in self.known_future_features:
                    future_out_dict[feature] = x[feature][:, self.index_bounds[0]:self.index_bounds[2]]
                else:
                    past_out_dict[feature] = x[feature][:, self.index_bounds[0]:self.index_bounds[1]]
                    future_out_dict[feature] = x[feature][:, :]

        future_out_dict = self.transform(future_out_dict)
        past_out_dict = self.transform(past_out_dict)

        out_dict = past_out_dict

        for f_key in future_out_dict.keys():

            if f_key in out_dict:
                future_out_dict[f_key] = future_out_dict[f_key][:, self.index_bounds[1]:self.index_bounds[2]]

                if isinstance(out_dict[f_key], torch.Tensor):
                    out_dict[f_key] = torch.cat([out_dict[f_key], future_out_dict[f_key]], dim=1)
                elif isinstance(out_dict[f_key], np.ndarray):
                    out_dict[f_key] = np.concatenate([out_dict[f_key], future_out_dict[f_key]], axis=1)
            else:
                out_dict[f_key] = future_out_dict[f_key]

        return out_dict

    def get_meta(self, x):
        x = {k:v for k,v in deepcopy(x).items() if k in self.meta_keys}
        out_dict = self.transform(x)
        print(out_dict.keys(), "FUCK")
        out_dict["timestamp"] = out_dict["timestamp"][:, self.index_bounds[0]:self.index_bounds[2]]
        return out_dict

    def __call__(self, x):

        """
        :param kwargs: dicitonary of inputs
        :return: dictionary of outputs
        """

        # get the tensors to device

        # set samples to device
        #x = self.sample_to_device(x)


        with torch.no_grad():
            # prefix with gt_ if these keys are in predicted columns
            # we want to take the first initial input periods


            input_dict = self.get_inputs(x)
            meta_dict = self.get_meta(x)



            sample_dict = {**input_dict, **meta_dict}


        return sample_dict


