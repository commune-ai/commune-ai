import os
import sys
import pandas as pd
import plotly.graph_objects as go
import numpy as np

class ExplainBlock(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def run(self,input_dict):
        baseline=1.0

        df = input_dict['df']

        datetime_list = df['datetime'].tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['roi'],
            fill='tonexty'))

        fig.update_layout(
            title="Return of Investment Over Time",
            yaxis_title="Return on Investment",
            xaxis_title="Date",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="white"
            )
        )
        return fig

