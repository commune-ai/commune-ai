
import os

import sys
from copy import deepcopy
import plotly
sys.path.append(os.environ['PWD'])
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
from commune.explain import BaseExplainProcess
 
 
class Process(BaseExplainProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config.meta.crypto.token/module.yaml"

    def process(self,**kwargs):
        if hasattr(self, 'explain') and not self.cfg.get('refresh'):
            if self.explain:
                return self.explain

        # sample_prediction(gt=input_dict[])
        df = self.get_data()

        self.explain = {}

        self.explain['Indicators (Abs)'] = self.plot_indicator_abs(df=df)
        self.explain['Indicators (Rel-Diff)'] = self.plot_indicator_diff(df=df)
        self.explain= self.get_json(self.explain)
        

    def plot_indicator_abs(self,df):

        fig_dict = {}

        for indicator in self.cfg['indicators']:
            fig = px.line(y=df[indicator], x=df['date'],
                        labels=dict(y=indicator, x='Date'))
            
            fig_dict[indicator] = fig


        return plot_bundle(fig_dict)



    def plot_indicator_diff(self,df):

        fig_dict = {}

        transform_pipeline =   SequentialTransformPipeline(pipeline=[low_pass_fft(lowest_freq_frac=0.5,buffer_period_frac=0.2), difference_transform()])
        
        for indicator in self.cfg['indicators']:
            diff = transform_pipeline.transform(df[indicator])
            color = ['green' if v>0 else 'red'for v in diff]
            fig = px.bar(y=transform_pipeline.transform(df[indicator]), x=df['date'], color=color,color_discrete_map="identity",
                        labels=dict(y=indicator, x='Date'))
            
            fig_dict[indicator] = fig


        return plot_bundle(fig_dict)

    
    @property
    def timestampBounds(self):

        end_datetime = datetime.datetime.utcnow() if self.cfg['end_time'] == 'utcnow' else \
                            datetime.datetime.fromisoformat(self.cfg['end_time'])

        start_datetime = end_datetime - datetime.timedelta(**self.cfg['look_back_period'])
        datetimeBounds = [start_datetime, end_datetime] 
        return list(map(lambda dt: dt.timestamp(), datetimeBounds))

    @property
    def tag(self):
        return self.token
    
    @property
    def token(self):
        return self.module['process'].cfg['token']
    @property
    def table_name(self):
        return self.module['process'].table_name('processed')

    def get_data(self):
        query = f'''
            
                SELECT 
                    *
                FROM {self.table_name}
                WHERE timestamp > {self.timestampBounds[0]} AND extension = 0 
                ORDER BY timestamp
                
            '''.replace("'", '"')

        df = self.client['postgres'].query(query, True)
        df['date'] = df['timestamp'].apply(lambda ts: datetime.datetime.fromtimestamp(ts))
        return df

if __name__ == '__main__':
    explain_block = Process.deploy(actor=False)
    explain_block.run()

    for k,v in explain_block.get_plot().items():
        st.write(v)

