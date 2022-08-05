import requests
import json
import datetime
import pandas as pd
import os
import sys
import bittensor
import streamlit as st
sys.path[0] = os.environ['PWD']
from commune.config import ConfigLoader
from commune.utils.misc import  chunk, dict_put
import ray
import random
import torch
from copy import deepcopy
# function to use requests.post to make an API call to the subgraph url
from commune.process import BaseProcess
from tqdm import tqdm

from commune.ray.utils import kill_actor, create_actor
from ray.util.queue import Queue
import itertools
from commune.process.extract.crypto.utils import run_query
from commune.plot.dag import DagModule
from commune.streamlit import StreamlitPlotModule

class BitModule(BaseProcess):
    
    default_cfg_path=f"bittensor.module"
    force_sync = False
    _NETWORKS = ['nakamoto', 'nobunaga', 'local']
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
        if sample_n == None:
            sample_n = self.cfg.get('sample_n', 100)


        if sample_mode == 'rank':
            metric=kwargs.get('rank', 'ranks')
            descending = kwargs.get('descending', True)
            sampled_uid_indices = self.argsort_uids(metric=metric, descending=descending)
        elif sample_mode == 'random':
            sampled_uid_indices = torch.randperm(self.n)[:sample_n]
        else:
            raise NotImplementedError

        sampled_graph_state = {}
        for k,v in graph_state.items():
            if len(v.shape)==0:
                continue

            if v.shape[0] == self.n:
                sampled_graph_state[k] = v[sampled_uid_indices]
            else:
                sampled_graph_state[k] = v
            
        return sampled_graph_state



    def sync_graph(self):
        self.graph.sync()
        # once the graph syncs, set the block
        self.block = self.graph.block.item()
        self.graph_state = self.graph.state_dict()


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
    def graph_params(self):
        return self.graph_state.keys()

    
    def st_metrics(self):
        cols = st.columns(3)
        cols[0].metric("Synced Block", bt.block, f'{-bt.blocks_behind} Blocks Behind')
        cols[1].metric("Current Block", bt.current_block, "-8%")
        cols[2].metric("Humidity", "86%", "4%")


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

        nodes=[dict(id=f"uid-{i}", 
                                label=f"uid-{i}", 
                                color='blue',
                                size=20) for i in range(4000)]
        
        edges = [() for i in range(100)]
        # for e in range(10000):
        #     edges = [dict]
        self.plot.dag.build(nodes=nodes, edges=edges)
    
    
    def plot_sandbox(self):
        self.plot.run(data=self.graph.to_dataframe())
    
    def st_sidebar(self):
        # self.block = st.sidebar.slider('Block', 0, )

        default_network_index = 0
        for i, n in enumerate(self._NETWORKS):
            if  n == self.DEFAULT_NETWORK:
                default_network_index = i
        self.network = st.sidebar.selectbox('Network', self._NETWORKS, default_network_index)


if __name__ == '__main__':
    
    st.sidebar.write('# Bittensor')
    bt = BitModule.deploy(actor=False)

    # st.sidebar.write(torch.nonzero(bt.graph.weights).numpy())
    # st.write(bt.graph.uids.shape)

    with st.sidebar.expander('Graph Params'):
        st.write(bt.describe_graph_state())

    st.write(bt.graph.block.item())
    # st.write(bt.graph.uids[])
    bt.st_metrics()
    # bt.view_graph()
    import random
    
    # st.write(bt.graph.ranks)
    # st.write(torch.nonzero(bt.graph.weights).shape)
    
    # st.write(bt.graph.to_dataframe())

    # with st.expander('Plot Sandbox'):
    #     bt.plot_sandbox()


    # with st.expander('dataframe'):
    #     
    




    # print(ray.get(bt.process.remote()))
