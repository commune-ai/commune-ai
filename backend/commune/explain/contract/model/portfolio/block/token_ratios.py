import os
import sys
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from commune.utils.misc import round_sig
from commune.process import BaseProcess

class ExplainBlock(BaseProcess):
    def run(self,input_dict):
        contract = input_dict['contract']
        state = contract['Trader'].state().dict()
        state['tokens'] = {}
        for tokenSymbol in contract['Trader'].getTokens():
            state['tokens'][tokenSymbol] = contract['Trader'].tokenStates(tokenSymbol).dict()
   
        tokenRatios = {k:v['ratio']/10000 for k,v in state['tokens'].items()}
        fig = go.Figure(data=[go.Pie(labels=list(tokenRatios.keys()), values=list(tokenRatios.values()), hole=.6)])

        marketValue = round_sig(state['marketValue']/10**18, 3)

        fig.update_layout(
            title=dict(text="Portfolio Token Ratio and Value", font_size=20, x=0.5),
            annotations=[dict(text='Market Value',y=0.61, font_size=25, showarrow=False),
                        dict(text='ETH',y=0.40, font_size=20, showarrow=False),
                        dict(text=f'{marketValue}', y=0.50, font_size=35, showarrow=False)],
            autosize=False,
            width=500,
            height=500

            )
        return fig