
import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math


if os.getenv('PWD') not in sys.path:
    sys.path.append(os.getenv('PWD'))
from commune.explain import BaseExplainProcess
from plotly.subplots import make_subplots
import datetime
import json
import torch
from commune.utils.misc import chunk, get_object, dict_get, torch_batchdictlist2dict


class ExplainProcess(BaseExplainProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config/explain/block/regression/crypto/past_samples.yaml"
        
    def get_subplots(self, subplot_list):

        self.token_fig = {}
        plot_count  = len(subplot_list)
        rows = int(math.ceil(plot_count ** 0.5))
        cols = int(math.ceil(plot_count / rows)) 
        fig = make_subplots(rows=rows, cols=cols)

        for i, sub_fig in enumerate(subplot_list):
            row =int(math.ceil((i+1)/cols))
            col = i%cols+1

            for trace in sub_fig.data:
                fig.add_trace(trace, row=row, col=col )


        fig.update_yaxes(nticks=5)
        fig.update_xaxes(nticks=5)
        fig.update_layout(
                            autosize=False,
                            width=1000,
                            height=1000)
        return fig


    def process(self,**kwargs):

        
        token_pred_list,  token_gt_list = self.get_data()
        
        token_fig_list = {}
        for token, gt_list in  token_gt_list.items():
            token_fig_list[token] = []
            for i,gt in enumerate(gt_list):
                pred = token_pred_list[token][i]
                fig = self.get_sample_plot(pred=pred, gt=gt)
                fig.update_yaxes(showticklabels=False)
                fig.update_xaxes(showticklabels=False)
                fig.add_layout_image(
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

                token_fig_list[token].append(fig)

            self.explain[token] = self.get_subplots(subplot_list=token_fig_list[token])
        self.explain = self.get_json(self.explain)
        return self.explain
    @staticmethod
    def get_sample_plot(gt = {'mean': [], 'time': []},
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
        fig.add_shape(type="rect",
                x0=gt['time'][-1], x1=pred['time'][-1],
                line=dict(
                    color="RoyalBlue",
                    width=2
                ),
                fillcolor="LightSkyBlue")

        return fig



    @property
    def timestamp_step(self):
        return int(self.cfg['timescale'][:-1])*60

    @property
    def period(self):
        return {k:v*self.timestamp_step for k,v in self.cfg['period'].items()}
    
    @property
    def step_size(self):
        return  self.cfg['step_size']*self.timestamp_step
    
    @property
    def timestamps(self):
        output = list(range(self.timestampBounds[0] + self.period['input'], self.timestampBounds[1]-self.step_size - self.period['output'] ,self.step_size))
        return output
    
    @property
    def timestampBounds(self):
        bounds =  [int(datetime.datetime.fromisoformat(self.cfg['start_date']).timestamp()),
              int(datetime.datetime.fromisoformat(self.cfg['end_date']).timestamp())]

        return bounds


    def get_data(self):

        token_pred_list = {k:[] for k in self.cfg['tokens']}
        token_gt_list = {k:[] for k in self.cfg['tokens']}
        query = f'''
            {{            
                prediction(
                    tokens: {self.cfg['tokens']},
                    timestamps: {self.timestamps},
                    timescale: "{self.cfg['timescale']}",
                    experiment: {self.cfg['experiment']}
                    updatePipeline: false)
                {{

                    token
                    lower
                    upper
                    mean
                    timestamp

                }}
            }}'''.replace("'", '"')
        pred_list= self.client['graphql'].query(query=query)['prediction']

        for i,pred in enumerate(pred_list):
            token = pred['token']
            pred['time'] = pred['timestamp']
            for k in ['mean', 'lower', 'upper']:
                pred[k] = (pred['gtPast'][-1]*torch.tensor(pred[k])).tolist()
            token_pred_list[token] += [{
                                    'time': pred['time'],
                                    'mean': pred['mean'],
                                    'lower': pred['lower'],
                                    'upper': pred['upper']}]

            token_gt_list[token] += [{
                'time':  list(map(lambda ts: datetime.datetime.fromtimestamp(ts), pred['timestampPast']+pred['timestamp'])),
                'mean': pred['gtPast'] + pred['gt']
            }]

        return token_pred_list, token_gt_list

def __name__ == '__main__':
    ExplainBlock.deploy().run()