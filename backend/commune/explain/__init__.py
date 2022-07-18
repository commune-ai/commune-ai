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

    def change_state(self):
        for mode in ['read', 'write']:
            self.cfg[mode]['explain']['params']['meta']['tag'] = self.tag
        # self.module.setup()

    @property
    def tag(self):
        return self.cfg.get('tag')


    @staticmethod
    def get_json(input_dict={}):
        def plot2jsonstr(v):
            if isinstance(v, go.Figure):
                v = v.to_json()
            return v
        return dict_fn(input_dict, plot2jsonstr)
 
    @staticmethod
    def get_plot(input_dict={}):
        if not input_dict:
            input_dict = deepcopy(self.explain)

        def filter_fn(x):
            if isinstance(x, str):
                x = json.loads(x)
            if 'data' in x and isinstance(x, dict):
                return plotly.graph_objects.Figure(x)
            else:
                return x

        return dict_fn(input_dict,
                                 filter_fn)

