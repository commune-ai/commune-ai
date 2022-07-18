
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
from commune.model.wrapper import MonteCarloDropout
import ray
from .hyperparams import hyperparams

class CompleteModel(RegressionBaseModel):
    default_cfg_path = f"{os.getenv('PWD')}/commune/model/regression/nbeats/base/model.yaml"
    model_name = 'nbeats'
    hyperparams = hyperparams
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.connect_data()
        self.build_model()
        self.define_metrics()
        self.resolve_device()


    def build_model(self): 
        cfg = self.cfg
        self.model = {}
        self.optimizer = {}

        for block_key in cfg['nbeats']['blocks'].keys():
            cfg['nbeats']['blocks'][block_key]['backcast_length'] = cfg['periods']['input']
            cfg['nbeats']['blocks'][block_key]['forecast_length'] = cfg['periods']['output']

        cfg['nbeats']['output_dim'] = 1

        self.targets = cfg['predicted_columns']
        cfg['nbeats']['targets'] = self.targets
        cfg['nbeats']['output_dim'] = 1

        self.model = NBEATS_Multivariate(**cfg['nbeats'])
        self.optimizer =  torch.optim.Adam([
                                                 {'params': self.model.parameters()}],
                                                **cfg['optimizer'])

    def forward(self, *args ,**kwargs):

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
                pivot_value = kwargs[target][:, self.cfg['periods']['input']-1, None]
                out_dict[f"pred_{time_mode}_{target}"] = pred_dict[time_mode][...,target_idx, 0] + pivot_value

        return out_dict

    def predict(self, **kwargs):
        if self.training:
            out_dict = {f"{k}-mean":v for k,v in self(**kwargs).items()}
        else:
            mc_dropout = MonteCarloDropout(model=self,
                                           **self.cfg['inference']['mc_dropout'])
            
            out_dict = mc_dropout(**kwargs)

        return out_dict

