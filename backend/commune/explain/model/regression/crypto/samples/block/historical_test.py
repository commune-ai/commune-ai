
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
from commune.process import BaseProcess
from plotly.subplots import make_subplots
import datetime
import json
import torch
from commune.utils.misc import chunk, get_object, dict_get, torch_batchdictlist2dict


class ExplainBlock(BaseProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config/explain/regression/crypto/block/historical_test.yaml"
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.cfg = cfg
        
    
    def process(self,**kwargs):

        
        token_pred,  token_gt = self.get_data()
        
        token_fig = {}
        for token, gt in  token_gt.items():
            pred = token_pred[token]
            fig = self.generate_plot(pred=pred, gt=gt)
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
        st.write(fig)
        # sample_prediction(gt=input_dict[])
        return fig
    def get_signal_color(self, signal):

        if signal > 0 :
            return 'green'
        elif signal < 0 :
            return 'red'
        else: 
            return 'yellow'
    def generate_plot(self,gt = {'mean': [], 'time': []},pred= {},
                        showlegend=False):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=gt['time'],
                                        y=gt['mean'],
                                        line=dict(color='rgb(15, 109, 153)'),
                                        name="Ground Truth",
                                        showlegend=showlegend
                                        ))

        fig.add_trace(go.Scatter( y=pred['future_gt'], x=pred['future_time'], 
                                mode='markers',
                                marker=dict(size=8,
                                            color=[ self.get_signal_color(s) for s in pred['profit']],
                                            )))






        return fig
    @property
    def timestamps(self):
        step_size = self.cfg['step_size']*3600
        output = list(range(self.timestampBounds[0], self.timestampBounds[1]-step_size,step_size))
        return output
    @property
    def timescale_period(self):
        return 60*int(self.cfg['timescale'][:-1])

    
    @property
    def timestampBounds(self):
        bounds =  [int(datetime.datetime.fromisoformat(self.cfg['start_date']).timestamp()),
              int(datetime.datetime.fromisoformat(self.cfg['end_date']).timestamp())]

        return bounds

    @property
    def timestamp(self):
        return int(datetime.datetime.utcnow().timestamp())

    def get_pred_signal(self, pred, threshold=0.000):
        if pred>1.0+threshold:
            return 1
        elif pred<1.0-threshold:
            return -1
        else:
            return 0


    def get_data(self):
        time_format = "%Y-%m-%dT%H:%M"
        timescale = self.cfg['timescale']
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
                    timestampPast

                }}
            }}'''.replace("'", '"')
        token_pred_df = {k:[] for k in self.cfg['tokens']}
        token_gt_df = {k:[] for k in self.cfg['tokens']}

        pred_list= self.client['graphql'].query(query=query)['prediction']

        for i,pred in enumerate(pred_list):
            token = pred['token']
            pred['time'] =  pred['timestamp']

            for j in range(len(pred['mean'])):
                token_pred_df[token] += [{
                                        'current_time': pred['timestampPast'][-1] ,
                                        'future_time': pred['time'][j],
                                        'signal': self.get_pred_signal(pred['mean'][j])
                                        }]
    
    
        for token in self.cfg['tokens']:
            st.write(token)
            query = f'''
                {{ 
                    sushiswap(
                                token: "{token}",
                                columns: {["token0Price", "timestamp", 'direction', 'reserveUSD', 'reserve0', 'reserve1']},
                                timestampMin: {self.timestampBounds[0]},
                                timestampMax: {self.timestampBounds[1]}
                                )

                }}'''.replace("'", '"')

            df = pd.read_json(json.loads(self.client['graphql'].query(query=query)['sushiswap']))

            df['time'] = df['timestamp'].apply(lambda dt: int(dt.to_pydatetime().timestamp()))
            token_index = df['direction'].max()
            df[f'tokenPriceUSD'] = (df[f'reserveUSD']*0.5)/df[f'reserve{token_index}']
            
            gt = {'mean':df['tokenPriceUSD'].tolist(),
                'time': df['time'].tolist()}

            token_pred_df[token] = pd.DataFrame(token_pred_df[token])
            current_gt_idx = torch.argmin(torch.abs(torch.tensor([gt['time']])-torch.tensor(token_pred_df[token]['current_time'].tolist()).unsqueeze(1) ), dim=1).tolist()
            future_gt_idx = torch.argmin(torch.abs(torch.tensor([gt['time']])-torch.tensor(token_pred_df[token]['future_time'].tolist()).unsqueeze(1) ), dim=1).tolist()

            token_pred_df[token]['current_gt'] = torch.tensor(gt['mean'])[current_gt_idx].tolist()
            token_pred_df[token]['future_gt'] = torch.tensor(gt['mean'])[future_gt_idx].tolist()
            token_pred_df[token]['gt_diff'] = token_pred_df[token]['future_gt'] - token_pred_df[token]['current_gt']
            token_pred_df[token]['profit'] = token_pred_df[token]['gt_diff']* token_pred_df[token]['signal']

            token_pred_df[token]['future_time'] = token_pred_df[token]['future_time'].apply(lambda ts: datetime.datetime.fromtimestamp(ts))
            token_pred_df[token]['current_time'] = token_pred_df[token]['current_time'].apply(lambda ts: datetime.datetime.fromtimestamp(ts))

            token_gt_df[token] = pd.DataFrame(gt)
            token_gt_df[token]['time'] = token_gt_df[token]['time'].apply(lambda ts: datetime.datetime.fromtimestamp(ts))


        return token_pred_df, token_gt_df


    def get_json(self):
        token_json_fig = {}
        fig = self.token_fig
        json_obj_str = json.dumps(fig.to_json())
        json_obj_str = json_obj_str.replace("'", '"')
        json_obj_str = json_obj_str.replace('True',  'true').replace('False', 'false')
        return json_obj_str


ExplainBlock.deploy().run()