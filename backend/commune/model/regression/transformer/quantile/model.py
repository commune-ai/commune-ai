
from functools import partial
import torch
from commune.utils.misc import get_object
from commune.model.block.transformer import Time2VecTransformer
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel
import ray

class CompleteModel(RegressionBaseModel):
    model_name = "transformer"
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = {}
        self.optimizer = {}

        cfg['transformer']['periods'] = cfg['periods']
        self.targets = cfg['predicted_columns']
        cfg['transformer']['targets'] = self.targets

        # Quantile Stuff
        assert 'quantiles' in cfg
        self.quantiles = cfg['quantiles']
        self.semantic_quantile_map = cfg['semantic_quantile_map']
        # ensure all of the quantiles are within the quantile map
        assert(all(sq in self.quantiles for sq in self.semantic_quantile_map.values()))
        cfg['transformer']["output_dim"] = len(self.quantiles)

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
                pred_key = f"pred_{time_mode}_{target}"
                for q_idx, q in enumerate(self.quantiles):
                    pivot = kwargs[target][:, self.cfg['periods']['input'] - 1, None]

                    # prediction key for the quantile
                    pred_q_key = f"{pred_key}_Q-{q}"
                    out_dict[pred_q_key] = pred_dict[time_mode][...,target_idx, q_idx] + pivot

                for q_key, q_val in self.semantic_quantile_map.items():
                    out_dict[f'{pred_key}-{q_key}'] = out_dict[f"{pred_key}_Q-{q_val}"]
        return out_dict
