import requests
import json
import datetime
import pandas as pd
import os
import sys
from copy import deepcopy
import ray
if os.environ['PWD'] not in sys.path:
    sys.path.append(os.environ['PWD'])
from commune.config import ConfigLoader
from commune.client.manager import ClientManager
from commune.config import ConfigLoader
from commune.ray import ActorBase
from commune.utils.misc import dict_get, dict_put, get_object, dict_has
# function to use requests.post to make an API call to the subgraph url
from commune.ray.utils import create_actor


class BaseProcess(ActorBase):
    """
    Manages loading and processing of data

    pairs: the pairs we want to consider
    base: the timeframe we consider buys on
    higher_tfs: all the higher timeframes we will use
    """
    defualt_cfg_path = None
    default_client_cfg_path = f'{os.environ["PWD"]}/commune/client/manager.yaml'
    default_client_cfg = ActorBase.load_config(default_client_cfg_path)
    config_loader = ConfigLoader(load_config=False)
    module = {}

    cache = {} # for caching things
    def __init__(
            self, 
            cfg, 
            ):

        super().__init__(cfg)

        self.cfg=cfg
        self.connect_clients()
        self.get_sub_modules()

        self.setup()
        self.resolve_queue_config()
        self.template_cfg = deepcopy(self.cfg)
        
    def setup(self):
        pass

    def resolve_queue_config(self):
        for mode in ['in', 'out']:
            queue_keys = ['queue', mode]
            queue_topic = f"{self.cfg['module']}.{mode}"
            if not dict_has(self.cfg, keys=queue_keys):
                dict_put(self.cfg, keys=queue_keys, value=queue_topic)

    def override(self, override={}):
        """
        overiide config with override dictionary
        
        """
        for k,v in override.items():
            dict_put(input_dict=self.__dict__,keys=k, value=v)

    def get_sub_modules(self):
        if 'sub_module' in self.cfg:
            assert isinstance(self.cfg['sub_module'], dict)

            for module_key,module_cfg in self.cfg['sub_module'].items():
                if isinstance(module_cfg, dict):
                    if 'cfg' in module_cfg:

                        '''
                        cfg: config
                        override: config override
                        '''
                        # module key must be a string 

                        actor_kwargs = module_cfg.get('actor', {})
                        override_kwargs = module_cfg.get('override', {})

                        if not actor_kwargs:
                            override_kwargs['client'] = self.client_manager


                        module = self.get_module(cfg=module_cfg['cfg'], 
                                                actor=actor_kwargs,
                                                override=override_kwargs)

                    elif  'module' in module_cfg:
                        '''
                        module: string of where module is located

                        '''

                        # module key must be a string 
                        assert isinstance(module_cfg['module'],str)

                        actor_kwargs = module_cfg.get('actor', {})
                        override_kwargs = module_cfg.get('override', {})

                        if not actor_kwargs:
                            override_kwargs['client'] = self.client_manager

                        module = self.get_module(cfg=module_cfg['module'],
                                                 actor= actor_kwargs, 
                                                 override=override_kwargs)
   
                elif isinstance(module_cfg, str):
                    if self.actor_exists(module_cfg):

                        '''
                        if the module is a ray actor, then get the handle 
                        note that you need to call remote() for every function
                        '''
                        module = self.get_actor(module_cfg)
                    else:
                        
                        module = self.get_module(cfg=module_cfg, override ={'client': self.client_manager})   

                # inserts the module based on the key of the module in sub_module config
                dict_put(input_dict=self.__dict__,
                                    keys= module_key,
                                    value= module)

    def change_state(self):
        pass

    def object_override(self):
        if self.object_dict.get('override'):
            self.override(self.object_dict['override'])


    def get_trigger_queue(self, item=None, mode='in', block=True): 

        
        if item == None:
            item  = deepcopy(self.object_dict)


        queue_path = ['queue',mode]


        topic = dict_get(self.object_dict, queue_path)
        if not isinstance(topic, bool) and bool(topic):
            topic = dict_get(self.cfg, keys=queue_path)
        # if the topic is false

        if bool(topic):
            if mode == 'in':
                return  self.client['ray'].queue.get(topic=topic, block=block)
            elif mode == 'out':

                return  self.client['ray'].queue.put(topic=topic, item=item , block=block)
            else:
                ('BROOOOO, the mode can only be in or out')
    
        return item

    def resolve_object(self, **kwargs):
        self.object_dict = kwargs
        self.object_override()

        self.object_dict = self.get_trigger_queue(mode='in')
        self.object_override()




    def preprocess(self, **kwargs):
        self.cfg = deepcopy(self.template_cfg)
        self.resolve_object(**kwargs)
        self.change_state()
        self.read_state()

    def run(self, **kwargs):
        self.preprocess(**kwargs)
        self.process(**self.object_dict)
        self.postprocess(**self.object_dict)
        return self.object_dict
        
    def postprocess(self, **kwargs):
        self.get_explain()
        self.write_state()   
        self.last_run_timestamp = datetime.datetime.utcnow().timestamp()
        self.get_trigger_queue(item=self.object_dict, mode='out')

    @property
    def last_run_delay(self):
        current_timestamp = datetime.datetime.utcnow().timestamp()
        if not hasattr(self, "last_run_timestamp"):
            return current_timestamp
        return current_timestamp - self.last_run_timestamp

    def run_again(self, threshold=3600):
        return self.last_run_delay > threshold




    @staticmethod
    def default_clients(clients =[], cfg=None):
        if cfg is None:
            cfg =  BaseProcess.load_config(BaseProcess.default_client_cfg_path)
        if len(clients)>0 :
            cfg['client'] = {client: cfg['client'][client] for client in clients}

        return BaseProcess.get_module(cfg=cfg).client


    def connect_clients(self):
        
        self.cfg['client'] = self.cfg.get('client')
        if isinstance(self.cfg['client'], dict):
            self.client_manager = self.get_module(cfg=self.cfg['client'])
            self.client = self.client_manager.client
        else:
            self.client_manager = self.cfg['client']
            self.cfg['client'] = self.client_manager.cfg['client']
            self.client = self.client_manager.client


    def process(self, **kwargs):
        raise NotImplementedError("Implement this shizzz fam")

    def read_state(self):
        if 'read' not in self.cfg:
            return
        for object_key, store_dict in self.cfg['read'].items():

            # only avoid reading state if you are refreshing the object or have an ignore flag

            if store_dict.get('refresh', self.cfg.get('refresh')):
                continue
            elif store_dict.get('ignore'):
                continue


            object_value = None
            object_value = self.client_manager.load(client=store_dict['client'],
                                                        params=store_dict['params'])
            
            if object_value:
                dict_put(self.__dict__, object_key, object_value)
            else:
                if not dict_has(self.__dict__, object_key):
                    dict_put(self.__dict__, object_key, store_dict.get('fallback_value', object_value))

    


    def write_state(self):
        if 'write' not in self.cfg:
            return
        

        for object_key, store_dict in self.cfg['write'].items():
            if  store_dict.get('ignore'):
                continue

            assert hasattr(self,object_key.split('.')[0]), f"{object_key} not found in self"
            object_value = dict_get(self.__dict__,object_key)
            self.client_manager.write(data=object_value,client=store_dict['client'],
                                                                params=store_dict['params'])


    def get_explain(self):
        if 'explain' in self.cfg:
            explain_cfg = deepcopy(self.cfg['explain'])
            if dict_has(explain_cfg, 'sub_module.process'):
                del explain_cfg['sub_module']['process']
            explain_module = self.get_module(explain_cfg)
            explain_module.run(override={'module.process': self})
            self.cfg['explain'] = explain_module.cfg

    def cancel_loop(self):
        self.loop_running = False
        self.loop_count= False

    def loop(self,**kwargs):
        self.loop_count = 0
        self.loop_running = True
        while self.loop_running:
            self.run(**kwargs)
            self.loop_count += 1
