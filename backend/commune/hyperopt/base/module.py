from commune.utils.misc import get_object
from copy import deepcopy
from commune.process.base import BaseProcess
import os

from copy import deepcopy

def stringify_list(input_list):
    return list(map(str,input_list))

class BaseHyperopt(BaseProcess):
    default_cfg_path = f"{os.environ['PWD']}/commune/hyperopt/config.yaml"
    def __init__(self,cfg={}):
        super().__init__(cfg=cfg)


    def build(self, cfg):
        self.build_hyperparams(cfg=cfg)
        self.tun_config = self.hyperparams
    def build_hyperparams(self, cfg):
        self.hyperparams = {}
        assert isinstance(cfg, dict)
        self.config_parser(cfg=cfg, root_node='')


    def hyperparam_builder(self, cfg, root_node="model"):
        is_module =  isinstance(cfg, dict) and ('module' in cfg)

        if is_module:
            module_class = get_object(cfg['module'])
            
            if hasattr(module_class, 'hyperparams'):
                hyperparams = {f"{root_node}.{k}":v 
                                for k,v in module_class.hyperparams.items()}
                self.hyperparams.update(hyperparams)


    def config_parser(self, cfg, root_node=""):

        cfg = deepcopy(cfg)

        if isinstance(cfg, dict):
            self.hyperparam_builder(cfg=cfg, root_node=root_node)
            key_list = list(cfg.keys())
        elif isinstance(cfg, list):
            key_list = list(range(len(cfg)))
        else:
            return cfg

        for k in key_list:
            v = cfg[k]
            if type(v) in [list,dict]:
                new_root_node = ".".join(stringify_list([root_node,k]))  if len(root_node) > 0 else str(k)
                cfg[k] = self.config_parser(cfg=v,root_node=new_root_node )

    def run(self, cfg, train_job, num_samples=1):
        """
        cfg: configuration of experiment
        tune_config: config of hyperopt tune distributios
        train_job_fn
        """
        raise NotImplementedError