
import os
import sys

from functools import partial
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from commune.utils.misc import get_object
from commune.model.block.deep_ar import DeepAR
from commune.model.regression.base import RegressionBaseModel
from commune.model.metric import *
import ray
class CompleteModel(RegressionBaseModel):
    model_name = "deep_ar"
    def __init__(self, cfg, data):
        super().__init__()

        self.connect_data(cfg=cfg,data=data)

        self.cfg = cfg

        self.model = {}
        self.optimizer = {}


        """Iniitalize the GRU"""
        self.targets = cfg['predicted_columns']

        cfg['deep_ar']["targets"] = self.targets

        # get the distribution
        distribution =  cfg['deep_ar']['distribution']
        if isinstance(distribution, str):
            cfg['deep_ar']['distribution'] = get_object(f"model.distribution.{distribution}")

        self.distribution = cfg['deep_ar']["distribution"]

        self.model = DeepAR(**cfg['deep_ar'])
        self.optimizer =  torch.optim.Adam([
                                                 {'params': self.model.parameters()}],
                                                **cfg['optimizer'])

        self.define_metrics()

    def forward(self, **kwargs):

        out_dict = {}

        """Pass the output of the gp into the transformer"""
        batch_size = list(kwargs.values())[0].shape[0]


        kwargs['encoder_lengths'] = torch.full(size=(batch_size,),
                                               fill_value=self.cfg['periods']['input'],
                                               device=self.cfg['device']).long()

        kwargs['decoder_lengths'] = torch.full(size=(batch_size,),
                                               fill_value=self.cfg['periods']['output'],
                                               device=self.cfg['device']).long()

        # output distribution parameters

        pred_dict = self.model(kwargs)

        # get distirbutions from parameters

        for time_mode in ["future"]:
            for target in self.targets:
                dist_params = pred_dict[target]
                out_dict[f'pred_{time_mode}_{target}_distribution'] = self.distribution(dist_params)

                pred_quantiles = out_dict[f'pred_{time_mode}_{target}_distribution'].to_quantiles(quantiles=[0.2, 0.5, 0.8])

                pred_lower = pred_quantiles[...,0]
                pred_mean = pred_quantiles[...,1]
                pred_upper = pred_quantiles[...,2]

                pivot_value = kwargs[target][:, self.cfg['periods']['input']-1, None]
                out_dict[f"pred_{time_mode}_{target}-lower"] = pred_lower + pivot_value
                out_dict[f"pred_{time_mode}_{target}-mean"] = pred_mean + pivot_value
                out_dict[f"pred_{time_mode}_{target}-upper"] = pred_upper + pivot_value

            return out_dict


    @classmethod
    def from_data(cls, cfg, data):
        model_name = 'deep_ar'

        cfg['batch_size'] = ray.get(data.get.remote('batch_size'))

        cfg[model_name]['temporal_features'] = ray.get(data.get.remote('input_columns'))
        cfg[model_name]['known_future_features'] = ray.get(data.get.remote('known_future_features'))

        # get categorical features
        categorical_feature_info = ray.get(data.get.remote('categorical_feature_info'))
        categorical_features = list(categorical_feature_info['unique_values'].keys())

        cfg[model_name]['categorical_features'] = categorical_features

        default_embedding_size = cfg[model_name]['embedding_size']
        embedding_sizes = {category: (cardinality, default_embedding_size)
                           for category, cardinality in categorical_feature_info['unique_values_count'].items()}
        embedding_sizes.update(cfg[model_name]['embedding_sizes'])

        cfg[model_name]['embedding_sizes'] = embedding_sizes

        return cls(cfg=cfg)





