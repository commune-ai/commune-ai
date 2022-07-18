


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
    def __init__(self, cfg, model_dict={}, aggregator=lambda x: x.mean(0)):
        super().__init__()

        self.cfg = cfg
        self.targets = cfg['predicted_columns']

        self.model = nn.ModuleDict()

        self.aggregator=aggregator

        self.optimizer = {}

        for model_key, model in model_dict.items():
            self.model[model_key] = model
            if hasattr(model, 'optimizer'):
                if isinstance(model.optimizer, torch.optim.Optimizer):
                    self.optimizer[model_key] = model.optimizer
        """Iniitalize the GRU"""

        self.define_metrics()

    # this is a non-parametric model, so we need not instantiate the learning rate
    def learning_rate_scheduler(self, train_state):
        return train_state


    def predict(self, **kwargs):
        out_dict = {}

        for model_key, model in self.model.items():
            model_output_dict = model.predict(**kwargs)

            ## Assumption, model outputs a dictionary of tensors
            for tensor_key, tensor_value in model_output_dict.items():
                    if tensor_key in out_dict:
                        out_dict[tensor_key] += [tensor_value]
                    else:
                        out_dict[tensor_key] = [tensor_value]

        # concat the list of tensors for each output tensor key

        for k, v in out_dict.items():

            if len(v) == 1:
                out_dict[k] = v[0]
            else:
                if isinstance(v[0], torch.Tensor):
                    out_dict[k] = self.aggregator(torch.stack(v))

        return out_dict




    @classmethod
    def from_data(cls, cfg, data):


        model_dict= {}
        for model_key,model_cfg in cfg['model'].items():
            model_class = get_object(f"model.{model_cfg['model_type']}.{model_cfg['model_name']}.{model_cfg['prediction_type']}.CompleteModel")
            model_dict[model_key] = model_class.from_data(cfg=model_cfg, data=data)

        return cls(cfg=cfg,model_dict=model_dict)


    def learning_step(self, **kwargs):
        '''
        Calculate the learning step
        '''
        out_dict = self.predict(**kwargs)

        out_dict.update(kwargs)

        # calculate sample wise matrics
        sample_metrics = self.calculate_metrics(out_dict)

        for optim_key in self.optimizer.keys():
            self.optimizer[optim_key].zero_grad()

        sample_metrics["total_loss"].mean().backward(retain_graph=True)

        for optim_key in self.optimizer.keys():
            self.optimizer[optim_key].step()

        return out_dict,sample_metrics

        



        return out_dict,sample_metrics



