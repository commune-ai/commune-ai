
import os
import sys

from functools import partial
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from commune.utils.misc import get_object
from commune.model.block.nbeats import NBEATS_Multivariate
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel
import ray
class CompleteModel(RegressionBaseModel):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = {}
        self.optimizer = {}



        # get the distribution

        if isinstance(cfg['distribution'], str):
            cfg['distribution'] = get_object(f"model.distribution.{cfg['distribution']}")

        self.distribution = cfg["distribution"]

        for block_key in cfg['nbeats']['blocks'].keys():
            cfg['nbeats']['blocks'][block_key]['backcast_length'] = cfg['periods']['input']
            cfg['nbeats']['blocks'][block_key]['forecast_length'] = cfg['periods']['output']

        cfg['nbeats']['output_dim'] = len(self.distribution.distribution_arguments)

        self.targets = cfg['predicted_columns']
        cfg['nbeats']['targets'] = self.targets


        self.model = NBEATS_Multivariate(**cfg['nbeats'])
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


        pred_dict = {}
        pred_dict['past'],pred_dict['future'] = self.model(kwargs)

        # get distirbutions from parameters

        for time_mode in pred_dict.keys():
            for target_idx ,target in enumerate(self.targets):
                dist_params = pred_dict[time_mode][...,target_idx,:]
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
        model_name = 'nbeats'

        data_info = data.get_info.remote()

        cfg['periods'] = data_info['periods']
        cfg[model_name]['temporal_features'] = data_info['input_columns']
        cfg[model_name]['known_future_features'] = data_info['known_future_features']

        # get categorical featuresx
        categorical_feature_info = data_info['categorical_feature_info']
        categorical_features = list(categorical_feature_info['unique_values'].keys())
        cfg[model_name]['categorical_features'] = categorical_features
        default_embedding_size = cfg[model_name]['embedding_size']
        embedding_sizes = {category: (cardinality, default_embedding_size)
                           for category,cardinality in categorical_feature_info['unique_values_count'].items()}
        embedding_sizes.update(cfg[model_name]['embedding_sizes'])

        cfg[model_name]['embedding_sizes'] = embedding_sizes

        return cls(cfg=cfg)
