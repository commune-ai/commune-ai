
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


        for block_key in cfg['nbeats']['blocks'].keys():
            cfg['nbeats']['blocks'][block_key]['backcast_length'] = cfg['periods']['input']
            cfg['nbeats']['blocks'][block_key]['forecast_length'] = cfg['periods']['output']


        self.targets = cfg['predicted_columns']
        cfg['nbeats']['targets'] = self.targets


        # Quantile Stuff
        assert 'quantiles' in cfg
        self.quantiles = cfg['quantiles']
        self.semantic_quantile_map = cfg['semantic_quantile_map']
        # ensure all of the quantiles are within the quantile map
        assert(all(sq in self.quantiles for sq in self.semantic_quantile_map.values()))
        cfg['nbeats']["output_dim"] = len(self.quantiles)

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

                pred_key = f"pred_{time_mode}_{target}"
                pivot_index = self.cfg['periods']['input'] - 1
                pivot = kwargs[target][:, pivot_index, None]

                for q_idx, q in enumerate(self.quantiles):
                    pred_q_key = f"{pred_key}_Q-{q}"

                    # prediction key for the quantile
                    out_dict[pred_q_key] = pred_dict[time_mode][:,:,target_idx, q_idx] + pivot

                for q_key, q_val  in self.semantic_quantile_map.items():
                    out_dict[f'{pred_key}-{q_key}'] = out_dict[f"{pred_key}_Q-{q_val}"]

        return out_dict


    @classmethod
    def from_data(cls, cfg, data):
        model_name = 'nbeats'

        cfg[model_name]['temporal_features'] = ray.get(data.get.remote('input_columns'))
        cfg[model_name]['known_future_features'] = ray.get(data.get.remote('known_future_features'))

        # get categorical features
        categorical_feature_info = ray.get(data.get.remote('categorical_feature_info'))
        categorical_features = list(categorical_feature_info['unique_values'].keys())

        cfg[model_name]['categorical_features'] = categorical_features

        default_embedding_size = cfg[model_name]['embedding_size']
        embedding_sizes = {category: (cardinality, default_embedding_size)
                           for category,cardinality in categorical_feature_info['unique_values_count'].items()}
        embedding_sizes.update(cfg[model_name]['embedding_sizes'])

        cfg[model_name]['embedding_sizes'] = embedding_sizes

        return cls(cfg=cfg)
