import streamlit as st
import bittensor
import plotly 
import _thread


class BitModule:
    DEFAULT_NETWORK  = 'nobunaga'
    _NETWORKS = ['nakamoto', 'nobunaga', 'local']
    def __init__(self,
                 network='nobunaga',
                 block=None):
        # self.sync_network(network=network, block=block)
        self._network = network
        self._block = block
    @property
    def block(self):
        return self.subtensor.block


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



    @property
    def current_block(self):
        return self.subtensor.get_current_block()

    @property
    def n(self):
        return self.subtensor.max_n

    def sync(self,network=None,  block=None):
        # Fetch data from URL here, and then clean it up.
        if network == None:
            network = self.network
        if network != self.network:
            self.subtensor = self.get_subtensor(network=network)
        if block != self.block or not hasattr(self, 'graph'):
            self.graph = self.get_graph(subtensor=self.subtensor , block=block)


    
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

manual =  st.expander('bro')

bt  = BitModule()
bt.st_sidebar()
st.write(bt.graph.to_dataframe())
bt.sync()

st.write(bt.graph.addresses)





st.write()
# st.write(dir(subtensor))
# wallet = bittensor.wallet()
# st.write(dir(wallet))
# wallet.get_coldkey('saller101@')
# st.write(wallet.get_coldkey('saller101@'))
# st.write(wallet.coldkey)
# BitModule.describe('wallet')
# wallet = bittensor.wallet()
# st.write(dir(wallet))
# st.write(bittensor.cli().create_new_coldkey())
# st.write(wallet.new_coldkey(overwrite=True))