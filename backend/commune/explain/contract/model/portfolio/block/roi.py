
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
        roi_percent = round_sig((state['marketValue']/(float(state['depositValue'])+1E-10) -1)*100, 2)


        if roi_percent > 0 :
            roi_ratios = {
                            'roi': abs(roi_percent),

                            'non-roi':  100 - abs(roi_percent),

                        }
            colors = ['green', 'black']
        else:
        
            roi_ratios = {
                            'non-roi':  100 - abs(roi_percent),
                            'roi': abs(roi_percent),            

                        }
            
            colors = ['black','red']

        
        fig = go.Figure(data=[])

        color = 'green' if roi_percent > 0 else 'red' 
        circle_step = 0.0
        hole_fraction = 0.6
        fig.add_trace(
            go.Pie(labels=list(roi_ratios.keys()),
                         values=list(roi_ratios.values()), 
                         hole=hole_fraction,
                         direction= 'clockwise',
                         sort=False,
                         showlegend=False,
                         textinfo='none',
                         domain={'x':[circle_step, 1-circle_step], 'y':[circle_step,1-circle_step]},
                         marker={'colors':colors }))

        # hole_fraction = 0.8
        # hole_fraction =1- ((1-hole_fraction)*(1-2*circle_step))
        # circle_step = 0.0
        # fig.add_trace(
        #     go.Pie(labels=list(roi_ratios.keys()),
        #                  values=list(roi_ratios.values()), 
        #                  hole=hole_fraction,
        #                  domain={'x':[circle_step,1-circle_step], 'y':[circle_step,1-circle_step]},
        #                  marker={'colors':['green','black']}))


        fig.update_layout(
            title=dict(text="Portfolio Return on Investment", font_size=20, x=0.45),
            annotations=[dict(text='ROI',y=0.61, font_size=25, showarrow=False),
                        dict(text=f'{roi_percent}%', y=0.50, font_size=40, 
                            font_color="lime" if roi_percent>0 else "red", showarrow=False)],
            autosize=True,
            width=500,
            height=500,
            


            )
        return fig



