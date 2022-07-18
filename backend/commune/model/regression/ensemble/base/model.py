
import os
from functools import partial
import torch
import torch.nn as nn
from commune.utils.misc import get_object
from commune.model.regression.base import RegressionBaseModel
from commune.model.metric import compute_mse
#from ..combiners.linear import linear_combiner
import mlflow
import ray

class CompleteModel(RegressionBaseModel):
    def __init__(self, cfg, models):
        super().__init__()
        self.cfg = cfg
        self.targets = cfg['predicted_columns']

        #self.combiner = linear_combiner(**cfg['combiner'])
        self.combiner = torch.mean

        self.models = nn.ModuleList(models)

        self.optimizer =  torch.optim.Adam([{'params': self.parameters()}],
                                                **cfg['optimizer'])
        #
        self.define_metrics()
    def forward(self, **kwargs):

        model_pred_dict = {}

        for model in self.models:
            pred_dict = model.predict(**kwargs)
            for pred_k, pred_v in pred_dict.items():
                if isinstance(pred_v, torch.Tensor):
                    if pred_k in model_pred_dict:
                        model_pred_dict[pred_k].append(pred_v)
                    else:
                        model_pred_dict[pred_k] = [pred_v]


        model_pred_dict = {pred_k: torch.stack(pred_v_list, dim=-1)
                           for pred_k, pred_v_list in model_pred_dict.items()}
        ensemble_pred_dict = {k: self.combiner(v, dim=-1).squeeze(-1) for k, v in model_pred_dict.items()}


        return ensemble_pred_dict
