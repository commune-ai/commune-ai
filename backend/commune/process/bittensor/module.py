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
from copy import deepcopy
# function to use requests.post to make an API call to the subgraph url
from commune.process import BaseProcess
from tqdm import tqdm

from commune.ray.utils import kill_actor, create_actor
from ray.util.queue import Queue
import itertools
from commune.process.extract.crypto.utils import run_query




class BitModule(BaseProcess):
    default_cfg_path=f"{os.getenv('PWD')}/commune/process/bittensor/module.yaml"

    DEFAULT_NETWORK  = 'nobunaga'
    _NETWORKS = ['nakamoto', 'nobunaga', 'local']
    def __init__(self,
                 cfg=None, sync=False, **kwargs):
        BaseProcess.__init__(self, cfg) 
        # self.sync_network(network=network, block=block)
        self._network = cfg.get('network')
        self._block = cfg.get('block')
        self.sync(network=self.network, block=self.block)



    @property
    def block(self):
        if self._block is None:
            return self.current_block
        return self._block

    @block.setter
    def block(self, block):
        self.sync(network=self.network, block=self.block)
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
        self.sync(network=self.network, block=self.block)
        self._block = block

    @property
    def network(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self._network

    @network.setter
    def network(self, network):
        assert network in self._NETWORKS
        self.sync(network=network, block=self.block)

        self._network = network
        dict_put()

    @property
    def current_block(self):
        return self.subtensor.get_current_block()

    @property
    def n(self):
        return self.subtensor.max_n

    def sync(self,network=None,  block=None):
        # Fetch data from URL here, and then clean it up.

        network = network if network else self.network
        block = block if block else self.block

        if network != self.network:
            self.subtensor = self.get_subtensor(network=network)
        if block != self.block or not hasattr(self, 'graph'):
            self.graph = self.get_graph( block=block)

    def process(self, **kwargs):
        self.sync
        
    def switch_network(self, network:str):
        self.network = network


    def graph_state(self,mode:str='df'):
        if mode in ['df', 'pandas']:
            return self.graph.to_dataframe()
        elif mode in ['torch.state_dict']:
            return self.graph.state_dict()

    def get_graph(self,block=None , save=True):
        network =  self.network
        block =  self.block
        subtensor =  self.subtensor


        # Fetch data from URL here, and then clean it up.
        graph = bittensor.metagraph(network=network, subtensor=subtensor)
        graph.load(network=network)
        print(graph.block)
        graph.sync(block=block)
        if save:
            graph.save()
        return graph

    


    @staticmethod
    def get_subtensor( network='nakamoto', **kwargs):
        '''
        The subtensor.network should likely be one of the following choices:
            -- local - (your locally running node)
            -- nobunaga - (staging)
            -- nakamoto - (main)
        '''
        subtensor = bittensor.subtensor(network=network, **kwargs)
        return subtensor
    
    '''
        streamlit functions
    '''

    @staticmethod
    def describe(module =None, sidebar = True, detail=False, expand=True):
        
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
    def st_sidebar(self):
        # self.block = st.sidebar.slider('Block', 0, )

        default_network_index = 0
        for i, n in enumerate(self._NETWORKS):
            if  n == self.DEFAULT_NETWORK:
                default_network_index = i
        self.network = st.sidebar.selectbox('Network', self._NETWORKS, default_network_index)



def d3_graph_example():
    # Import library
    from d3graph import d3graph, vec2adjmat

    # Set source and target nodes
    source = ['node A','node F','node B','node B','node B','node A','node C','node Z']
    target = ['node F','node B','node J','node F','node F','node M','node M','node A']
    weight = [5.56, 0.5, 0.64, 0.23, 0.9, 3.28, 0.5, 0.45]

    # Create adjacency matrix
    adjmat = vec2adjmat(source, target, weight=weight)

    # target  node A  node B  node F  node J  node M  node C  node Z
    # source                                                        
    # node A    0.00     0.0    5.56    0.00    3.28     0.0     0.0
    # node B    0.00     0.0    1.13    0.64    0.00     0.0     0.0
    # node F    0.00     0.5    0.00    0.00    0.00     0.0     0.0
    # node J    0.00     0.0    0.00    0.00    0.00     0.0     0.0
    # node M    0.00     0.0    0.00    0.00    0.00     0.0     0.0
    # node C    0.00     0.0    0.00    0.00    0.50     0.0     0.0
    # node Z    0.45     0.0    0.00    0.00    0.00     0.0     0.0

    # Initialize
    d3 = d3graph()

    # Build force-directed graph with default settings
    d3.graph(adjmat)


    return d3.show()



if __name__ == '__main__':
    
    bt = BitModule.deploy(actor=False)

    st.write(bt)


    # print(ray.get(bt.process.remote()))
