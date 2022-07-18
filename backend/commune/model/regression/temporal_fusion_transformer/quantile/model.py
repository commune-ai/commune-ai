
from functools import partial
import torch
from torch import nn
from commune.model.block.temporal_fusion_transformer import TemporalFusionTransformer
from commune.model.metric import *
from commune.model.regression.base import RegressionBaseModel
import ray
class CompleteModel(RegressionBaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = {}
        self.optimizer = {}

        """Iniitalize the GRU"""

        self.targets = cfg['predicted_columns']
        cfg['temporal_fusion_transformer']["targets"] = self.targets

        # Quantile Stuff
        assert 'quantiles' in cfg
        self.quantiles = cfg['quantiles']
        self.semantic_quantile_map = cfg['semantic_quantile_map']
        # ensure all of the quantiles are within the quantile map
        assert(all(sq in self.quantiles for sq in self.semantic_quantile_map.values()))
        cfg['temporal_fusion_transformer']["output_size"] = len(self.quantiles)
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
            for time_mode in ["future"]:
                pred_key = f"pred_{time_mode}_{target}"
                for q_idx, q in enumerate(self.quantiles):

                    pivot = kwargs[target][:,self.cfg['periods']['input']-1, None]

                    # prediction key for the quantile
                    pred_q_key = f"{pred_key}_Q-{q}"
                    out_dict[pred_q_key] = pred_dict[pred_key][:,:, q_idx] + pivot

                for q_key, q_val  in self.semantic_quantile_map.items():
                    out_dict[f'{pred_key}-{q_key}'] = out_dict[f"{pred_key}_Q-{q_val}"]
        return out_dict

    def learning_step(self, **kwargs):
        out_dict = self(**kwargs)
        out_dict.update(kwargs)
        metrics = self.calculate_metrics(out_dict)

        self.optimizer.zero_grad()
        metrics["total_loss"].backward(retain_graph=True)
        self.optimizer.step()

        return out_dict,metrics




    @classmethod
    def from_data(cls, cfg, data):
        model_name = 'temporal_fusion_transformer'

        cfg['batch_size'] = ray.get(data.get.remote('batch_size'))

        cfg[model_name]['temporal_features'] = ray.get(data.get.remote('input_columns'))
        cfg[model_name]['known_future_features'] = ray.get(data.get.remote('known_future_features'))

        # get categorical features
        categorical_feature_info = ray.get(data.get.remote('categorical_feature_info'))
        categorical_features = list(categorical_feature_info['unique_values'].keys())

        cfg[model_name]['categorical_features'] = categorical_features

        default_embedding_size = cfg[model_name]['embedding_size']
        embedding_sizes = {category: (cardinality, default_embedding_size)
                           for category,cardinality in categorical_feature_info['unique_values_count'].items()}
        embedding_sizes.update(cfg[model_name]['embedding_sizes'])

        cfg[model_name]['embedding_sizes'] = embedding_sizes

        return cls(cfg=cfg)
