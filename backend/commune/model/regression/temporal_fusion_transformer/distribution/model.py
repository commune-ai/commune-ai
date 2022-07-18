
import os
import sys
import ray
from functools import partial
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from commune.utils.misc import get_object
from commune.model.block.temporal_fusion_transformer import TemporalFusionTransformer
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel

class CompleteModel(RegressionBaseModel):
    model_name = "transformer"
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = {}
        self.optimizer = {}


        """Iniitalize the GRU"""

        self.targets = cfg['predicted_columns']
        self.distribution = get_object(f"model.distribution.{cfg['distribution']}")

        cfg['temporal_fusion_transformer']["targets"] = self.targets
        cfg['temporal_fusion_transformer']["output_size"] = len(self.distribution.distribution_arguments)

        cfg['temporal_fusion_transformer']["device"] = cfg['device']

        self.model = TemporalFusionTransformer(**cfg['temporal_fusion_transformer'])
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


        pred_dict = self.model(kwargs)


        for target in self.targets:
            pivot_idx = self.cfg['periods']['input']-1
            pivot = kwargs[target][:, pivot_idx-1: pivot_idx]
            pred_key = f"pred_future_{target}"

            out_dict[f'{pred_key}_distribution'] = self.distribution(pred_dict[pred_key])
            pred_quantiles = out_dict[f'{pred_key}_distribution'].to_quantiles(quantiles=[0.2, 0.5, 0.8])

            out_dict[f"pred_future_{target}-lower"] = pred_quantiles[..., 0] + pivot
            out_dict[f"pred_future_{target}-mean"] = pred_quantiles[..., 1] + pivot
            out_dict[f"pred_future_{target}-upper"] = pred_quantiles[..., 2] + pivot

        return out_dict
