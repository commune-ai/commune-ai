
from functools import partial
import torch
import ray
from torch import nn
from commune.utils.misc import get_object
from commune.model.block.transformer import Time2VecTransformer
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel
from commune.model.wrapper import MonteCarloDropout
import ray

class CompleteModel(RegressionBaseModel):
    model_name = "transformer"
    def __init__(self, cfg, data):
        super().__init__()
        self.connect_data(cfg=cfg,data=data)
        self.model = {}
        self.optimizer = {}


        cfg['transformer']['periods'] = cfg['periods']
        cfg['transformer']['output_dim'] = 1
        self.targets = cfg['predicted_columns']
        cfg['transformer']['targets'] = self.targets


        self.model = Time2VecTransformer(**cfg['transformer'])
        self.optimizer =  torch.optim.Adam([
                                                 {'params': self.model.parameters()}],
                                                **cfg['optimizer'])


        self.define_metrics()

        self.mc_dropout = MonteCarloDropout(**cfg['inference']['mc_dropout'])


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

