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
from commune.explain import BaseExplainProcess

class ExplainModule(BaseExplainProcess):
    
    def __init__(
            self, 
            cfg,
    ):
        super().__init__(cfg=cfg)

    def run_explainers(self, input_dict):
        self.explain ={}
        for explainer_key, explainer in self.module['explain'].items():
            self.explain[explainer_key] = explainer.run(input_dict).to_json()
    
    
    def process(self, **kwargs):
        self.run_explainers(input_dict={'contract': self.module['process'].contract})


    
