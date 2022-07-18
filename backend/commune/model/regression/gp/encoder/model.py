


import os
import sys

from functools import partial
import torch
from torch import nn
from typing import Tuple, Dict
from copy import deepcopy
from commune.model.block.gp import RegressionEncoderGP
from commune.model.regression.base import RegressionBaseModel
from commune.model.block.transformer.block import MultiHeadedAttention
from commune.model.metric import *
from commune.utils.misc import get_object
from gpytorch.mlls import ExactMarginalLogLikelihood
import ray



class GP_Encoder(RegressionBaseModel):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = nn.ModuleDict()
        self.optimizer = {}



        """Setup the Encoder"""

        # setup encoder
        cfg['encoder']["periods"] = cfg['periods']
        encoder_kwargs = deepcopy(cfg['encoder'])
        # get the encoder class
        encoder_class = get_object(encoder_kwargs['module'])
        del encoder_kwargs['module'] # we dont incldue the module key in the kwargs
        # setup the encoder class
        self.model['encoder'] = encoder_class(**encoder_kwargs)

        self.encoder_mh_attn = MultiHeadedAttention(h=cfg['encoder_attn_heads'], d_model=cfg['encoder']['output_dim'])

        self.encoder_prediction_layer = nn.Linear(cfg['encoder']['output_dim'], 3)


        """
        Setup the Gaussian Process
        """

        # we want a multidimensional gaussian pipeline
        self.targets = cfg['predicted_columns']
        cfg['gp']["targets"] = self.targets
        cfg['gp']['batch_size'] = cfg['batch_size']
        cfg['gp']["device"] = cfg['device']
        cfg['gp']["periods"] = cfg['periods']
        cfg['gp']["input_dim"] = cfg['encoder']['output_dim']


        self.model['gp'] = RegressionEncoderGP(**cfg['gp'])


        # we just want to optimize the encoder at this abstraction

        self.optimizer =  torch.optim.Adam(list(self.model['encoder'].parameters()) +
                                           list(self.encoder_prediction_layer.parameters()),
                                                **cfg['optimizer'])

        self.learning_count = 0

        self.time_modes = ['future']
        if self.cfg['predict_past']:
            self.time_modes += ['past']


        self.define_metrics()


    def forward(self, **kwargs):
        self.learning_count += 1

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
        enc_dict = self.model['encoder'](kwargs, output_type=Dict)

        enc_dict = {k:torch.cat(torch.split(v.clone(),1,dim=-2), dim=0).squeeze(-2) for k,v in enc_dict.items()}

        enc_dict['future'] = self.encoder_mh_attn(query=enc_dict['future'],
                                                key=enc_dict['past'],
                                                value=enc_dict['past']
                                                  )

        pivot  = torch.cat([kwargs[t][:,self.cfg['periods']['input']-1, None, None] for t in self.targets], dim=0)
        enc_pred_dict = {k:self.encoder_prediction_layer(v) + pivot\
                         for k,v in enc_dict.items()}


        gp_kwargs = {
            'x_train':  enc_dict['past'].clone().detach(),
            'x_test':  enc_dict['future'].clone().detach(),
            'y_train':  torch.cat([kwargs[t][:,:self.cfg['periods']['input']]\
                                   for t in self.targets], dim=0)
        }

        gp_pred_dict = {}
        gp_pred_dict['past'], gp_pred_dict['future'] = self.model['gp'](**gp_kwargs)




        for target in self.targets:
            for time_mode in self.time_modes:

                pred_dict = {
                    'lower': gp_pred_dict[time_mode]['lower'] + enc_pred_dict[time_mode][...,0],
                    'mean': gp_pred_dict[time_mode]['mean'] + enc_pred_dict[time_mode][...,1],
                    'upper': gp_pred_dict[time_mode]['upper'] + enc_pred_dict[time_mode][...,2],

                }

                out_dict.update({f"pred_{time_mode}_{target}-{k}":v for k,v in pred_dict.items()})

        return out_dict



    @classmethod
    def from_data(cls, cfg, data):
        model_name = 'encoder'
        cfg['batch_size'] = ray.get(data.get.remote('batch_size'))
        cfg['periods'] = ray.get(data.get.remote('periods'))

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
