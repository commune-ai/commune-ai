
import os
import sys
import ray
from functools import partial
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from commune.utils.misc import get_object
from commune.model.block.rnn import AttentionRNN
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel
from commune.model.wrapper import MonteCarloDropout

class RNN_Base(RegressionBaseModel):
    model_name = "rnn"
    def __init__(self, cfg, data):
        super().__init__()
        self.connect_data(cfg=cfg,data=data)

        self.cfg = cfg

        self.model = {}
        self.optimizer = {}



        cfg['rnn']['periods'] = cfg['periods']

        self.targets = cfg['predicted_columns']
        cfg['rnn']['targets'] = self.targets
        cfg['rnn']['output_dim'] = 1


        self.model = AttentionRNN(**cfg['rnn'])
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
        pred_dict = self.model(kwargs)

        # get distirbutions from parameters

        for time_mode in pred_dict.keys():
            for target_idx ,target in enumerate(self.targets):
                out_dict[f"pred_{time_mode}_{target}"] = pred_dict[time_mode][...,target_idx, 0]

        return out_dict


    def predict(self, **kwargs):

        if self.training:
            out_dict = {f"{k}-mean":v for k,v in self(**kwargs).items()}
        else:
            mc_dropout = MonteCarloDropout(model=self,
                                           **self.cfg['inference']['mc_dropout'])

            out_dict = mc_dropout(**kwargs)

        return out_dict
   