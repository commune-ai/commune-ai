
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
from commune.explain import BaseExplainProcess

import math

from commune.transformation.block.torch import difference_transform, low_pass_fft
from commune.transformation.block.base import SequentialTransformPipeline


class Process(BaseExplainProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/explain/process/regression/crypto/sushiswap/sample_generator/module.yaml"
    def __init__(self, cfg):
        super().__init__(cfg=cfg)


    @property
    def tag(self):
        return self.module['process'].cfg['token']

    @property
    def timestamps(self):
        return [int(datetime.datetime.utcnow().timestamp())]*10

    @property
    def sample_count(self):
        return len(self.timestamps)
    

    def plot_sample_features(self,sample_dict ):

        fig_dict = {}


        for feature in self.module['process'].cfg['feature_group']['con']:
            
            plot_count  = self.sample_count
            rows = int(math.ceil(plot_count ** 0.5))
            cols = int(math.ceil(plot_count / rows)) 
            fig = make_subplots(rows=rows, cols=cols,
                                specs=[[{'secondary_y':True}]*cols]*rows)

            for i in range(self.sample_count):
                row =int(math.ceil((i+1)/cols))
                col = i%cols+1

                
                time = list(map(lambda ts: datetime.datetime.fromtimestamp(ts).isoformat(),
                            sample_dict['processed']['timestamp'][i]))
                fig.add_trace(go.Bar(y=list(sample_dict['processed'][feature][i]),
                                        x=time,
                                        name='processed: {feature}', showlegend=False,),
                                        
                                        secondary_y=False, row=row , col=col )

                
                fig.add_trace(go.Scatter(y=list(sample_dict['raw'][feature][i]),
                                        x=time,
                                        name='raw: {feature}', 
                                        showlegend=False
                                        ),
                                        
                                        secondary_y=True, row=row, col=col)
            fig.update_layout(width=1000, height=1000)

            fig_dict[feature] = fig


        return fig_dict

    def process(self,**kwargs):


        if hasattr(self, 'explain') and not self.cfg.get('refresh'):
            if self.explain:
                return self.explain

        # if self.module['process'].run_again(threshold=3600):
        #     print("AGAIN")
        #     self.module['process'].run()
        self.explain = {}
        timstamps = []
        self.sample_dict = self.get_sample_dict()
        self.explain['samples'] = self.plot_sample_features(self.sample_dict)
        self.explain = self.get_json(self.explain)

    @property
    def timestampBounds(self):

        end_datetime = datetime.datetime.utcnow() if self.cfg['end_time'] == 'utcnow' else \
                            datetime.datetime.fromisoformat(self.cfg['end_time'])

        start_datetime = end_datetime - datetime.timedelta(**self.cfg['look_back_period'])
        datetimeBounds = [start_datetime, end_datetime] 
        return list(map(lambda dt: dt.timestamp(), datetimeBounds))

    @property
    def token(self):
        return 'WETH'
        return self.module['process'].cfg['token']


    def get_sample_dict(self):

        sample_dict = {}
        sample_dict['raw'] = self.module['process'].get_batch(timestamps=self.timestamps, process_samples=False)
        sample_dict['processed'] = self.module['process'].get_batch(timestamps=self.timestamps, process_samples=True)
        return sample_dict

if __name__ == '__main__':
    ExplainBlock.deploy(actor=False).run()

