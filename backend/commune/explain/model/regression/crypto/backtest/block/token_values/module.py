import os
import sys
import pandas as pd
import plotly.graph_objects as go
import numpy as np

class ExplainBlock(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def run(self,input_dict, output_json=False):
        df = input_dict['df']

        token_values_df = pd.DataFrame(df['token_values'].tolist())
        datetime_list = df['datetime'].tolist()
        print(token_values_df)

        fig = go.Figure()

        cumulative_vector = np.array([0.0] * len(token_values_df)).astype(float)

        tokens = sorted(token_values_df.columns, reverse=True)

        for token in tokens:
            cumulative_vector += np.array(token_values_df[token].tolist()).astype(float)

            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=cumulative_vector,
                fill='tonexty',
                name=token))

        fig.update_layout(
            title="Token Portfolio Value over Time",
            yaxis_title="Value (ETH)",
            xaxis_title="Date",
            legend_title="Tokens",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="white"
            )
        )
        if output_json:
            fig = fig.to_json()

        return fig
