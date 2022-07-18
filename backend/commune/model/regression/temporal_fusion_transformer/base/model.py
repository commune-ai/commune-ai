

from functools import partial
import torch
from torch import nn
from commune.model.block.temporal_fusion_transformer import TemporalFusionTransformer
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel
from commune.model.wrapper import MonteCarloDropout
import ray
class CompleteModel(RegressionBaseModel):
    model_name = "temporal_fusion_transformer"
    def __init__(self, cfg, data):
        super().__init__()
        self.connect_data(cfg=cfg,data=data)

        self.model = {}
        self.optimizer = {}


        """Iniitalize the GRU"""

        self.targets = cfg['predicted_columns']

        cfg['temporal_fusion_transformer']["targets"] = self.targets
        cfg['temporal_fusion_transformer']["output_size"] = 1

        cfg['temporal_fusion_transformer']["device"] = cfg['device']

        self.model = TemporalFusionTransformer(**cfg['temporal_fusion_transformer'])
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


        output_dict = self.model(kwargs)



        for target in self.targets:
            pivot_idx = self.cfg['periods']['input']-1
            pivot = kwargs[target][:, pivot_idx-1: pivot_idx, None]

            pred_key = f"pred_future_{target}"
            out_dict[pred_key] = (output_dict[pred_key] + pivot).squeeze(-1)


        return out_dict


    def predict(self, **kwargs):

        if self.training:
            out_dict = {f"{k}-mean":v for k,v in self(**kwargs).items()}
        else:
            mc_dropout = MonteCarloDropout(model=self,
                                           **self.cfg['inference']['mc_dropout'])

            out_dict = mc_dropout(**kwargs)

        return out_dict
