import ray
from commune.config import ConfigLoader
from commune.ray.utils import create_actor, actor_exists, kill_actor
from commune.utils.misc import dict_put, get_object, dict_get
import os
import datetime
from types import ModuleType

class ActorBase: 
    config_loader = ConfigLoader(load_config=False)
    default_cfg_path = None
    def __init__(self, cfg):
        self.cfg = cfg 
        self.start_timestamp = datetime.datetime.utcnow().timestamp()

    @staticmethod
    def load_config(cfg=None, override={}, local_var_dict={}):
        """
        cfg: 
            Option 1: dictionary config (passes dictionary) 
            Option 2: absolute string path pointing to config
        """
        return ActorBase.config_loader.load(path=cfg, 
                                    local_var_dict=local_var_dict, 
                                     override=override)

    @classmethod
    def default_cfg(cls, override={}, local_var_dict={}):

        return cls.config_loader.load(path=cls.default_cfg_path, 
                                    local_var_dict=local_var_dict, 
                                     override=override)

    @staticmethod
    def get_module(cfg, actor=False, override={}):
        """
        cfg: path to config or actual config
        client: client dictionary to avoid child processes from creating new clients
        """
        if isinstance(cfg,type):
            return cfg

        process_class = None
        if isinstance(cfg, str):
            # check if object is a path to module, return None if it does not exist
            process_class = ActorBase.get_object(key=cfg, handle_failure=True)

        if isinstance(process_class, type):
            cfg = process_class.default_cfg()
        else:

            cfg = ActorBase.load_config(cfg)
            assert isinstance(cfg, dict)
            assert 'module' in cfg
            process_class = ActorBase.get_object(cfg['module'])

                

        return process_class.deploy(cfg=cfg, override=override, actor=actor)

    @staticmethod
    def get_object(key, prefix = 'commune', handle_failure= False):

        return get_object(path=key, prefix=prefix, handle_failure=handle_failure)

    @classmethod
    def deploy(cls, cfg=None, actor=False , override={}, local_var_dict={}):
        """
        deploys process as an actor or as a class given the config (cfg)
        """
        if cfg == None:
            cfg = cls.default_cfg_path

        cfg = cls.load_config(cfg=cfg, 
                             override=override, 
                            local_var_dict=local_var_dict)




        if actor:
            if isinstance(actor, dict):
                cfg['actor'].update(actor)
            elif isinstance(actor, bool):
                pass
            else:
                raise Exception('Only pass in dict (actor args), or bool (uses cfg["actor"] as kwargs)')  
            return cls.deploy_actor(cfg=cfg, **cfg['actor'])

        elif 'spawn' in cfg.get('actor', {}):
            del cfg['actor']['spawn']
            return cls.deploy_actor(cfg=cfg, **cfg['actor'])
        else:
            return cls(cfg=cfg)

    @classmethod
    def deploy_actor(cls,
                        cfg,
                        name='actor',
                        detached=True,
                        resources={'num_cpus': 1, 'num_gpus': 0.1},
                        max_concurrency=1,
                        refresh=False,
                        verbose = True, 
                        redundant=False):
        return create_actor(cls=cls,
                        name=name,
                        cls_kwargs={'cfg': cfg},
                        detached=detached,
                        resources=resources,
                        max_concurrency=max_concurrency,
                        refresh=refresh,
                        return_actor_handle=True,
                        verbose=verbose,
                        redundant=redundant)

    def get(self, key):
        return self.getattr(key)

    def getattr(self, key):
        if hasattr(self, key.split('.')[0]): 
            obj = getattr(self, key.split('.')[0])
            rest_of_keys = '.'.join(key.split('.')[1:])
            if len(rest_of_keys)>0:
                obj =  dict_get(input_dict=obj, keys=rest_of_keys)
            return obj

    def down(self):
        self.kill_actor(self.cfg['actor']['name'])

    @staticmethod
    def kill_actor(actor):
        kill_actor(actor)
    
    @staticmethod
    def actor_exists(actor):
        return actor_exists(actor)

    @staticmethod
    def get_actor(actor_name):
        return ray.get_actor(actor_name)

    @property
    def context(self):
        if self.actor_exists(self.actor_name):
            return ray.runtime_context.get_runtime_context()

    @property
    def actor_name(self):
        return self.cfg['actor']['name']
    
    @property
    def module(self):
        return self.cfg['module']

    @property
    def name(self):
        return self.cfg.get('name', self.module)
