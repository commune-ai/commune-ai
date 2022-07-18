import os
from symbol import return_stmt
import sys
sys.path[0] = os.environ['PWD']
import datetime
from copy import deepcopy
import pandas as pd
from commune.utils.misc import get_object
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import json
from commune.process import BaseProcess



class ExplainModule(BaseProcess):
    def __init__(
            self, 
            cfg,
    ):
        
        # self.client = cfg['client']
        super().__init__(cfg=cfg)
        self.explain ={}
        self.explain_figure = {}
        self.build_explainers()
        


    def build_explainers(self):
        self.explainers = {}
        
        for explainer_key, explainer_cfg in self.cfg['explainers'].items():
            self.explainers[explainer_key] = get_object(explainer_cfg['module']+'.ExplainBlock')(explainer_cfg)

    def run_explainers(self, input_dict):
        
        for explainer_key, explainer in self.explainers.items():
            self.explain_figure[explainer_key] = explainer.run(input_dict)
            self.explain[explainer_key] = json.dumps(explainer.run(input_dict).to_json()).replace("'", '"').replace('True',  'true').replace('False', 'false')
    def change_state(self):

        for mode in ['read','write']:
            for explain_key in self.explain_keys:
                self.cfg[mode][f'explain.{explain_key}'] =  deepcopy(self.cfg[mode]['explain.{explain_key}'])
                self.cfg[mode][f'explain.{explain_key}']['params']['meta'] = {'name': explain_key, 
                                                                                        'taxonomy': self.taxonomy}
               
            del self.cfg[mode]['explain.{explain_key}']
    
    
    
    def process(self, **kwargs):
        self.run_explainers(input_dict=kwargs)

    @property
    def explain_keys(self):
        return self.cfg['explainers'].keys()
    
    @property
    def taxonomy(self):
        return self.cfg['taxonomy']

    def get_explainer(self, explainer_key, output_json=True):
        fig  = self.explain_figure[explainer_key]
        if output_json:
            return fig.to_json()

        return fig
    def get_explainers(self, explainer_keys=[], output_json=True):
    
        if len(explainer_keys) == 0 :
            explainer_keys = self.explain_keys

        return [ self.get_explainer(explainer_key, output_json)
                   for explainer_key in explainer_keys ]
    
