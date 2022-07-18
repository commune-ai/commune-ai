
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math

if os.getenv('PWD') not in sys.path:
    sys.path.append(os.getenv('PWD'))
from commune.process import BaseProcess
from plotly.subplots import make_subplots
import datetime
import streamlit as st
import json
import torch

class ExplainBlock(BaseProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config/explain/model/regression/crypto/block/current_samples.yaml"
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        
    
    def process(self,**kwargs):

        
        token_pred,  token_gt = self.get_data()
        token_fig = {}
        for token, gt in  token_gt.items():
            pred = token_pred[token]
            fig = self.sample_prediction(pred=pred, gt=gt)
            token_fig[token] = fig

        plot_count  = len(token_fig)
        rows = int(math.ceil(plot_count ** 0.5))
        cols = int(math.ceil(plot_count / rows)) 
        fig = make_subplots(rows=rows, cols=cols)
        # st.write(rows, cols)

        for i, (token, sub_fig) in enumerate(token_fig.items()):
            row =int(math.ceil((i+1)/cols))
            col = i%cols+1

            for trace in sub_fig.data:
                fig.add_trace(trace, row=row, col=col )

            fig.add_layout_image(
                row=row,
                col=col,
                source=self.cfg['token_logo_map'][token],
                xref="x domain",
                yref="y domain",
                x=0.05,
                y=0.95,
                xanchor="left",
                yanchor="top",
                sizex=0.3,
                sizey=0.3,
            )
   
        self.token_fig = fig
        fig.update_yaxes(nticks=5)
        fig.update_xaxes(nticks=5)
        fig.update_layout(
                        autosize=False,
                        width=1000,
                        height=1000)



        # sample_prediction(gt=input_dict[])
        return fig
    @staticmethod
    def sample_prediction (gt = {'mean': [], 'time': []},
                        pred = {'mean':[], 'upper':[], 'lower': [], 'time': []},
                        showlegend=False):
        fig = go.Figure()
        period = {}

        fig.add_trace(go.Scatter(x=gt['time'],
                                        y=gt['mean'],
                                        line=dict(color='red'),
                                        name="Ground Truth",
                                        showlegend=showlegend
                                        ))

        fig.add_trace(go.Scatter(x=pred['time'],
                                            y=pred['mean'],
                                            line=dict(color='rgba(39, 215, 180, 1.0)'),
                                            name="Mean",
                                            showlegend=showlegend
                                            ))


        fig.add_trace(go.Scatter(x=gt['time'],
                                y=gt['mean'],
                                line=dict(color='red'),
                                name="Ground Truth",
                                showlegend=showlegend
                                        ))

        fig.add_trace(go.Scatter(x=pred['time'],
                                        y=pred['mean'],
                                        line=dict(color='rgba(39, 215, 180, 1.0)'),
                                        name="Mean Prediction",
                                        showlegend=showlegend
                                        ))

        fig.add_trace(go.Scatter(x=pred['time'],
                                        y=pred['upper'],
                                        line=dict(color='rgba(39, 215, 180, 0.5)'),
                                        showlegend=showlegend
                                        ))

        fig.add_trace(go.Scatter(x=pred['time'],
                                        y=pred['lower'],
                                        line=dict(color='rgba(39, 215, 180, 0.5)'),
                                        fillcolor='rgba(39, 215, 180, 0.1)',
                                        fill='tonexty',
                                        showlegend=showlegend

                                        ))
        return fig
    @property
    def timestamp(self):
        return int(datetime.datetime.utcnow().timestamp())

    @property
    def timescale_period(self):
        return 60*int(self.cfg['timescale'][:-1])
    @property
    def timestampBounds(self):
        bounds =  [int(self.timestamp-self.cfg['period']['input']*self.timescale_period),
                int(self.timestamp + self.cfg['period']['output']*self.timescale_period)]

        return bounds
    @property
    def timestamp(self):
        return int(datetime.datetime.utcnow().timestamp())
    def table_name(self, token):
        return self.cfg['load']['processed_data']['params']['table_name'].format(token=token,
                                                                base_ticker=self.cfg["base_ticker"],
                                                                processed_data_tag="base")


    def get_prediciton(self):
        token_pred = {}
        input_period = self.cfg['period']['input']
        output_period = self.cfg['period']['output']
        timescale = self.cfg['timescale']
        query = f'''
            {{            
                prediction(
                    tokens: {self.cfg['tokens']},
                    timestamps: {[self.timestamp]},
                    timescale: "{self.cfg['timescale']}",
                    updatePipeline: false)
                {{

                    token
                    lower
                    upper
                    mean
                    datetime

                }}
            }}'''.replace("'", '"')
        pred_list= self.client['graphql'].query(query=query)['prediction']
        for pred in pred_list:
            token = pred['token']
            pred['time'] = list(map(lambda dt: datetime.datetime.fromisoformat(dt).strftime("%m:%d:%H:%M"), pred['datetime']))
            del pred['token'], pred['datetime']
            token_pred[token] = pred    

        return token_pred    
    def get_data(self):

        token_pred = self.get_prediciton() 
        token_gt = {}
        for token in self.cfg['tokens']:
            
            df = self.client['postgres'].query(
                f'''
                SELECT
                    timestamp,
                    "tokenPriceUSD"

                FROM {self.table_name(token=token)}
                WHERE timestamp >= {self.timestampBounds[0]} AND 
                      timestamp <= {self.timestampBounds[1]}
                ORDER BY "timestamp" ''',
                output_pandas=True)
            

            df['time'] = df['timestamp'].apply(lambda ts: datetime.datetime.fromtimestamp(ts).strftime("%H:%M"))
            gt = {'mean':df['tokenPriceUSD'].tolist(),
                'time': df['time'].tolist()}
            
            for k in ['upper', 'lower', 'mean']:
                token_pred[token][k] = list(torch.tensor(token_pred[token][k])*gt['mean'][-1])

            token_gt[token] = gt



        return token_pred, token_gt
    def get_json(self):
        token_json_fig = {}
        fig = self.token_fig
        json_obj_str = json.dumps(fig.to_json())
        json_obj_str = json_obj_str.replace("'", '"')
        json_obj_str = json_obj_str.replace('True',  'true').replace('False', 'false')
        return json_obj_str

if __name__ == '__main__':
    explain_block = ExplainBlock.deploy()
    explain_block.run()