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
from commune.utils import format_token_symbol
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
        print("getter of x called")
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
        print("setter of x called")
        self.sync(network=network, block=self.block)

        self._network = network
        dict_put()



    @property
    def current_block(self):
        return self.subtensor.get_current_block()

    @property
    def n(self):
        return self.subtensor.max_n

    def sync(self,network='nakamoto',  block=None):
        # Fetch data from URL here, and then clean it up.
        if network != self.network:
            self.subtensor = self.get_subtensor(network=network)
        if block != self.block or not hasattr(self, 'graph'):
            self.graph = self.get_graph(subtensor=self.subtensor , block=block)


    def process(self, **kwargs):
        print(self.graph)
    
    def graph_state(self,mode:str='df'):
        if mode == in ['df', 'pandas', da
        return self.graph.to_dataframe()

    def get_graph(self, network=None,block=None,  subtensor=None , save=True):
        # Fetch data from URL here, and then clean it up.
        graph = bittensor.metagraph(network=network, subtensor=subtensor)
        graph.load(network=network)

        # graph.sync(block=block)
        if save:
            graph.save(network=network)
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


if __name__ == '__main__':

    with ray.init(address='auto', namespace='serve'):
        bt = BitModule.deploy(actor=True)

        # print(ray.get(bt.process.remote()))
