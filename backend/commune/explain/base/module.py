import os
import sys
sys.path.append(os.environ['PWD'])
from copy import deepcopy
import plotly
from commune.process import BaseProcess
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import datetime
import streamlit as st
import json
import torch
from commune.utils.plot import plot_bundle
from commune.utils.misc import dict_fn
import math

from commune.transformation.block.torch import difference_transform, low_pass_fft
from commune.transformation.block.base import SequentialTransformPipeline


class BaseExplainProcess(BaseProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config/explain/process/regression/crypto/sushiswap/sample_generator/module.yaml"

    explain = {}
    def change_state(self):
        for mode in ['read', 'write']:
            self.cfg[mode]['explain']['params']['meta']['tag'] = self.tag
        # self.module.setup()

    @property
    def tag(self):
        return self.cfg.get('tag')


    @staticmethod
    def plot2json(plot, to_string=False):
        '''
        Converts plot into explain
        '''
        if isinstance(plot, go.Figure):
            plot_json = plot.to_json()
            if to_string:
                plot_json = json.dumps(plot)
        return plot_json


    @staticmethod
    def json2plot(plot_json, mode='plotly'):
        '''
        converts json to plot (only plotly supported)
        '''
        from_string = isinstance(plot_json,str)

        __supported_modes__ = ['plotly']

        if mode in 'plotly':
            if from_string:
                plot_json = json.loads(plot_json)
            assert isinstance(plot_json, dict)
            return plotly.graph_objects.Figure(plot_json)
        else:
            raise Exception(f'only, {__supported_modes__} is supproted')

    @staticmethod
    def get_json(input_dict={}):
        return dict_fn(input_dict, plot2jsonstr)
 
    def get_plot(self, input_dict={}):
        if not input_dict:
            #
            input_dict = deepcopy(self.explain)

        def filter_fn(x):
            if isinstance(x, str):
                x = json.loads(x)
            if isinstance(x, dict) and 'data' in x:
                return plotly.graph_objects.Figure(x)
            else:
                return x

        return dict_fn(input_dict,
                                 filter_fn)


    def put_explain(self,key,plot, to_json=True):
        if to_json:
            plot = self.plot2json(plot)
        dict_put(input_dict=self.explain, keys=key, value=plot)

    def get_explain(self,key, to_json=False):
        return self.explain[key]


    def list_explain(self):
        return list(self.explain.keys())

    def rm_explain(self,key):
        self.explain.pop(key)

    def refresh_explain(self):
        self.explain = {}

