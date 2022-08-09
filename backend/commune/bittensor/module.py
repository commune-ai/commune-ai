
import os
import sys
sys.path[0] = os.environ['PWD']
import numpy as np
import math
import requests
import json
import datetime
import pandas as pd
import bittensor
import streamlit as st
import plotly.express as px
from commune.config import ConfigLoader
from commune.utils.misc import  chunk, dict_put, round_sig
import ray
import random
import torch
from copy import deepcopy
# function to use requests.post to make an API call to the subgraph url
from commune.process import BaseProcess
from tqdm import tqdm
from plotly.subplots import make_subplots
from commune.ray.utils import kill_actor, create_actor
from ray.util.queue import Queue
import itertools
from commune .process.extract.crypto.utils import run_query
# from commune.plot.dag import DagModule
from commune.streamlit import StreamlitPlotModule, row_column_bundles





class BitModule(BaseProcess):
    sample_n = 400
    sample_mode = 'rank'
    sample_metric = 'ranks'
    sample_descending = True
    
    default_cfg_path=f"bittensor.module"
    force_sync = False
    def __init__(self,
                 cfg=None, sync=False, **kwargs):
        BaseProcess.__init__(self, cfg) 
        # self.sync_network(network=network, block=block)
        self._network = cfg.get('network')
        self._block = cfg.get('block')
        self.sync()
        self.plot = StreamlitPlotModule()



    @property
    def block(self):
        if self._block is None:
            return self.current_block
        return self._block

    @block.setter
    def block(self, block):
        self._block = block


    @property
    def subtensor(self):
        if not hasattr(self,'_subtensor'):
            self._subtensor = self.get_subtensor(network=self.network)
        return self._subtensor

    @subtensor.setter
    def subtensor(self, subtensor):
        self._subtensor = subtensor


    @block.setter
    def block(self, block):
        self._block = block

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        assert network in self._NETWORKS
        if network != self.network:
            self.sync()
            self._network = network
            

    @property
    def current_block(self):
        return self.subtensor.get_current_block()

    @property
    def n(self):
        return self.subtensor.max_n

    def sync(self, force_sync=False):
        self.force_sync = force_sync
        # Fetch data from URL here, and then clean it up.
        self.get_subtensor()
        self.get_graph()

    def process(self, **kwargs):
        self.sync
        
    def switch_network(self, network:str):
        self.network = network


    def graph_state(self,mode:str='df'):
        if mode in ['df', 'pandas']:
            return self.graph.to_dataframe()
        elif mode in ['torch.state_dict']:
            return self.graph.state_dict()

    @property
    def graph_path(self):
        return f'backend/{self.network}B{self.block}.pth'


    def load_graph(self):
        # graph_state_dict = self.client['minio'].load_model(path=self.graph_path) 
        # self.graph.load_from_state_dict(graph_state_data)

        self.graph.load()
        self.block = self.graph.block.item()
        if not self.should_sync_graph:
            self.set_graph_state()
        


    def set_graph_state(self, sample_n=None, sample_mode='rank', **kwargs):
        graph_state = self.graph.state_dict()
        self.graph_state =  self.sample_graph_state(graph_state=graph_state, sample_n=sample_n, sample_mode=sample_mode, **kwargs)

    def sample_graph_state(self, graph_state , sample_n=None,  sample_mode='rank', **kwargs ):
        '''
        Args:
            sample_mode: 
                the method of sampling the neurons data. There are two 

        '''

        sample_n = sample_n if sample_n != None else self.sample_n
        sample_mode = sample_mode if sample_mode != None else self.sample_mode
        metric=kwargs.get('metric', self.sample_metric)
        descending = kwargs.get('descending', self.sample_descending)

        if sample_mode == 'rank':
            sampled_uid_indices = self.argsort_uids(metric=metric, descending=descending)[:sample_n]
        elif sample_mode == 'random':
            sampled_uid_indices = torch.randperm(self.n)[:sample_n]
        else:
            raise NotImplementedError


        self.sampled_uid_indices = sampled_uid_indices

        sampled_graph_state = {}
        for k,v in graph_state.items():
            if len(v.shape)==0:
                continue
            elif (len(v.shape) == 1 and v.shape[0] == self.n) or k in ['endpoints'] :
                sampled_graph_state[k] = v[sampled_uid_indices]
            elif len(v.shape) == 2 and v.shape[0] == self.n and v.shape[1] == self.n:
                sampled_graph_state[k] = v[sampled_uid_indices]
                sampled_graph_state[k] = sampled_graph_state[k][:, sampled_uid_indices]
            else:
                sampled_graph_state[k] = v
            
        return sampled_graph_state

    def sync_graph(self):
        self.graph.sync()
        # once the graph syncs, set the block
        self.block = self.graph.block.item()
        self.set_graph_state()


    def save_graph(self):
        # graph_state_dict = self.graph.state_dict()

        # graph_state_data = self.client['minio'].save_model(path=self.graph_path,
        #                                                     data=graph_state_dict) 
  
        # self.graph.load_from_state_dict(graph_state_data)
        # self.block = self.graph.block.item()
        self.graph.save()
        
    def argsort_uids(self, metric='rank', descending=True):
        prohibited_params = ['endpoints', 'uids', 'version']

        if metric in prohibited_params:
            return None

        metric_tensor = getattr(self.graph, metric, None)

        if metric_tensor == None :
            return None
        else:
            metric_shape  = metric_tensor.shape
            if len(metric_shape) == 2 and metric_shape[0] == self.n:
                metric_tensor = torch.einsum('ij->i', metric_tensor)
            if metric_shape[0] == self.n:
                return torch.argsort(metric_tensor, descending=descending, dim=0).tolist()    
    @property
    def should_sync_graph(self):
        return (self.blocks_behind > self.cfg['blocks_behind_sync_threshold']) or self.force_sync

    @property
    def blocks_behind(self):
        return self.current_block - self.block

    def get_graph(self):

        # Fetch data from URL here, and then clean it up.
        self.graph = bittensor.metagraph(network=self.network, subtensor=self.subtensor)
        self.load_graph()
        if self.should_sync_graph:
            self.sync_graph()
            self.save_graph()


    def get_subtensor(self, network='nakamoto', **kwargs):
        '''
        The subtensor.network should likely be one of the following choices:
            -- local - (your locally running node)
            -- nobunaga - (staging)
            -- nakamoto - (main)
        '''
        self.subtensor = bittensor.subtensor(network=network, **kwargs)
    
    '''
        streamlit functions
    '''


    def adjacency(mode='W', ):
        return torch.nonzero(self.graph.weights)

    def describe_graph_state(self, shape=True):
        return {k:dict(shape=v.shape, type=v.dtype ) for k,v in  self.graph_state.items()}


    @property
    def graph_state_params(self):
        return self.graph_state.keys()

    def agg_param(self, param='rank', agg='sum', decimals=2):
        param_tensor = getattr(self.graph, param)
        return round(getattr(torch,agg)(param_tensor).item(), decimals)


    @property
    def networks(self):
        return bittensor.__networks__

    def st_select_network(self):

        with st.sidebar.form('Sync Network Options'):
            network2idx = {n:n_idx for n_idx, n in  enumerate(self.networks)}
            default_idx = network2idx['nakamoto']
            network = st.selectbox('Select Network', self.networks,default_idx)
            block = st.number_input('Select Block', 0,self.current_block, self.block)
            submitted = st.form_submit_button("Sync")


        # ''')

    def st_metrics(self):

        cols = st.columns(3)
        # self.block = st.sidebar.slider('Block', 0, )
        cols[0].metric("Synced Block", f'{self.block}', f'{-self.blocks_behind} Blocks Behind')
        cols[1].metric("Network", 'nakamoto')
        cols[2].metric("Active Neurons ", f'{self.n}/{self.n}')


        metrics = [ 'trust', 'stake', 'consensus']
        fn_list = []
        fn_args_list = []
        from copy import deepcopy
        for metric in metrics:
            metric_show = 'Total '+ metric[0].upper() + metric[1:].lower()
            # st_fn = 
            metric_value = self.agg_param(metric)
            st.write()
            fn_args_list.append([metric_show, metric_value])
            fn_list.append(lambda name, value: st.metric(name, value ))

        row_column_bundles(fn_list= fn_list, fn_args_list=fn_args_list, cols_per_row=3)

    @classmethod
    def describe(cls, module =None, sidebar = True, detail=False, expand=True):
        
        if module is None:
            module = cls

        _st = st.sidebar if sidebar else st
        st.sidebar.markdown('# '+str(module))
        fn_list = list(filter(lambda fn: callable(getattr(module,fn)) and '__' not in fn,  dir(module)))
        
        
        def content_fn(fn_list=fn_list):
            fn_list = _st.multiselect('fns', fn_list)
            for fn_key in fn_list:
                fn = getattr(module,fn_key)
                if callable(fn):
                    _st.markdown('#### '+fn_key)
                    _st.write(fn)
                    _st.write(type(fn))
        if expand:
            with st.sidebar.expander(str(module)):
                content_fn()
        else:
            content_fn()

    @property
    def n(self):
        return self.subtensor.n

    def view_graph(self):

        n= 100
        edge_density = 0.1
        nodes=[dict(id=f"uid-{i}", 
                                label=f"uid-{i}", 
                                color='red',
                                size=100) for i in range(n)]

        st.write('bro')
        edges = torch.nonzero(torch.rand(n,n)>edge_density).tolist()


        edges = [dict(source=f'uid-{i}', target=f'uid-{j}') for i,j in edges]




        self.plot.dag.build(nodes=nodes, edges=edges)
    
    def st_describe_graph_state(self):
        with st.sidebar.expander('Graph Params'):
            st.write(self.describe_graph_state())

    def plot_sandbox(self):
        self.plot.run(data=self.graph.to_dataframe())
    
    def st_sidebar(self):
        bt_url = 'https://yt3.ggpht.com/9fs6F292la5PLdf-ATItg--4bhjzGfu5FlIV1ujfmqlS0pqKzGleXzMjjPorZwgUPfglMz3ygg=s900-c-k-c0x00ffffff-no-rj'
        bt_url = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAACoCAMAAABt9SM9AAAAVFBMVEUAAAD///8hISGxsbFgYGBcXFx1dXXn5+dUVFQrKysGBgYZGRmKiore3t6YmJinp6f39/fAwMDIyMhnZ2cQEBBsbGzv7+9BQUHX19fz8/N/f3/h4eHkVQerAAAA6klEQVR4nO3Yu27CQBRFURNjIAbCIwGc5P//M9gUCGi4Te4YrdV4yqNdjKWpKgAAAAAAAAAAAAAAAAAAAAAAAAB4IU32gIK1bzeWq3X2omLN5pN7H9mbSrV5SDXpsjeVqnlIta2zNxWr3jXTs8W+77Q6nxfZiwr22V6+dR/LX/A5Q6xp9oqRECtArACxAsQKECtArACxAsQKECtArIBrrK82e0vxhliH/nR8z95SvEMfa32qlrtj9pQR6IaHv/m3i+sZ3eWh9Cd7xzjM9r9d7cICAAAAAAAAAAAAAAAAAAAAAAAAAAAA4F/8AYjjA5tLvfqfAAAAAElFTkSuQmCC'
        st.sidebar.image(bt_url, width=300)
        # st.markdown('''
        # # BitDashboard
        # ---
        # ''')
        self.st_select_network()
        self.st_sample_params()
        self.st_describe_graph_state()

    def st_main(self):

        self.st_metrics()
        
        df = self.graph_df()
        with st.expander('Graph Dataframe'):
            st.write(df)
        plot_df = df.drop(['uid', 'active'], axis=1)
        self.st_distributions(df = plot_df)
        self.st_scatter(df=plot_df)
        self.st_relationmap()

        
        with st.expander('Custom Plots'):
            self.plot.run(data=plot_df)

    def st_relationmap(self):
        with st.expander('Relation Map'):
            metric = st.selectbox('Select a ', ['weights', 'bonds'], 0)
            z = self.graph_state[metric].tolist()
            fig = self.plot.imshow(z, text_auto=True, title=f'Relation Map of {metric.upper()}')
            fig.update_layout(autosize=True, width=1000, height=1000)
            st.write(fig)
    def st_distributions(self, df):
        plot_columns = [c for c in df.columns if c not in ['uid', 'active']]


        with st.expander('Distibutions'):
            fn_list = [lambda col: st.write(self.plot.histogram(df, x=col, title=f'Distribution of {col.upper()}', color_discrete_sequence=random.sample(px.colors.qualitative.Plotly,1)))]*len(plot_columns)
            fn_args_list = [[col,] for col in plot_columns]
            row_column_bundles(fn_list=fn_list, fn_args_list=fn_args_list)

    def st_scatter(self, df):
        plot_columns = ['stake', 'rank', 'trust', 'consensus', 'incentive', 'dividends', 'emission']
        
        with st.expander('Scatter'):
            fn_list = []
            fn_args_list = []
            for col_x in plot_columns:
                for col_y in ['rank']:
                    if col_x != col_y:

                        fn_list += [lambda col_x, col_y: st.write(self.plot.scatter(df, x=col_x, y=col_y, title=f'{col_x.upper()} vs {col_y.upper()}', color_discrete_sequence=random.sample(px.colors.qualitative.Plotly,1)))]
                        fn_args_list += [[col_x,col_y]]
        
            row_column_bundles(fn_list=fn_list, fn_args_list=fn_args_list)


    def st_run(self):
        
        st.set_page_config(layout="wide")
        self.st_sidebar()
        self.st_main()


    def graph_df(self):
        df_dict= {
                'uid': self.graph_state['uids'],
                'active': self.graph_state['active'],             
                'stake': self.graph_state['stake'],             
                'rank': self.graph_state['ranks'],            
                'trust': self.graph_state['trust'],             
                'consensus': self.graph_state['consensus'],             
                'incentive': self.graph_state['incentive'],             
                'dividends': self.graph_state['dividends'],          
                'emission': self.graph_state['emission']
            }


        return pd.DataFrame(df_dict)

    def st_sample_params(self):
        with st.sidebar.form("sample_n_form"):
            self.sample_n = st.slider('Sample N', 1, self.n, self.sample_n )
            self.sample_mode = st.selectbox('Sample Mode', ['rank', 'random'], 0 )
            self.sample_metric = st.selectbox('Sample Metric', ['ranks', 'trust'], 0 )
            self.sample_descending = 'descending'== st.selectbox('Sample Metric', ['descending', 'ascending'], 0 )
            submitted = st.form_submit_button("Sample")

            if submitted:
                self.set_graph_state()

if __name__ == '__main__':
    

    BitModule.deploy(actor=False).st_run()

    import random

    