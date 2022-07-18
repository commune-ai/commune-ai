


import os
import sys

from functools import partial
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from commune.utils.misc import get_object
from commune.model.block.gp import RegressionGP
from commune.model.regression.base import RegressionBaseModel
from commune.model.metric import *
from gpytorch.mlls import ExactMarginalLogLikelihood

import ray
class CompleteModel(RegressionBaseModel):
    model_name = "gp"
    def __init__(self, cfg, data):
        super().__init__()
        self.connect_data(cfg=cfg,data=data)

        self.model = nn.ModuleDict()
        self.optimizer = {}


        """Iniitalize the GRU"""
        self.targets = cfg['predicted_columns']

        # we want a multidimensional gaussian pipeline
        cfg['gp']["targets"] = self.targets
        cfg['gp']['batch_size'] = cfg['batch_size']
        cfg['gp']["device"] = cfg['device']
        cfg['gp']["periods"] = cfg['periods']

        self.model['gp'] = RegressionGP(**cfg['gp'])

        self.cfg = cfg

        self.define_metrics()

    # this is a non-parametric model, so we need not instantiate the learning rate
    def learning_rate_scheduler(self, train_state):
        return train_state

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

        pred_dict = self.model['gp'](kwargs)
        out_dict = {f"pred_future_Close-{k}":v for k,v in pred_dict.items()}

        return out_dict



    def learning_step(self, **kwargs):
        '''
        Calculate the learning step
        '''
        out_dict = self.predict(**kwargs)

        out_dict.update(kwargs)

        # calculate sample wise matrics
        sample_metrics = self.calculate_metrics(out_dict)


        return out_dict,sample_metrics
