
import os
import sys

import ray
from functools import partial
import torch
from torch import nn
from commune.utils.misc import get_object
from commune.model.block.transformer import Time2VecTransformer
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel


class CompleteModel(RegressionBaseModel):
    model_name = "transformer"
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = {}
        self.optimizer = {}

        # get the distribution

        if isinstance(cfg['distribution'], str):
            cfg['distribution'] = get_object(f"model.distribution.{cfg['distribution']}")
        self.distribution = cfg["distribution"]

        cfg['transformer']['periods'] = cfg['periods']
        cfg['transformer']['output_dim'] = len(self.distribution.distribution_arguments)
        self.targets = cfg['predicted_columns']
        cfg['transformer']['targets'] = self.targets


        self.model = Time2VecTransformer(**cfg['transformer'])
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
