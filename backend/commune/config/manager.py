
import os, sys
import ray
from copy import deepcopy
sys.path.append(os.environ['PWD'])
from commune.process import BaseProcess
from commune.utils.misc import dict_fn, dict_get, dict_has
import streamlit as st

class ConfigManager(BaseProcess):
    default_cfg_path =  f"{os.environ['PWD']}/commune/config/manager.yaml"

    def process(self, module, tags={}):
        """
        kwargs = {
            'module': query
        }

        """
        self.object_dict['config'] =  self.find_modules(module=module, tags=tags)
    def list_modules(self, query={}, select=[]):

        documents = self.find_modules(query=query, select=['module',*select])
        return documents

    def find_modules(self, query={}, select=[],  module=None):
        
        
        if module:
            query['module'] = module
        projection = {s:1 for s in select} if select else None
        cfg = self.client['mongo'].load(collection='config',
                                        database='commune', 
                                        query=query, 
                                        projection=projection,
                                       return_one=False, remove_id=True)
        cfg = dict_fn(cfg,self.resolve_pipeline_cfg)
        return cfg 
            
    def resolve_explan_ipfs(self, cfg):
        if dict_has(cfg, 'explain.write.explain.params'):
            ipfs_params = deepcopy(dict_get(cfg, 'explain.write.explain.params'))
            ipfs_params['return_hash'] =True
            cfg['explain']['hash'] = self.client['ipfs'].load(**ipfs_params)
        return  cfg


    def resolve_pipeline_cfg(self, cfg):
        
        if dict_has(cfg, 'dag'):
            for process_key, process_cfg_template in cfg['dag'].items():

                if 'template' in process_cfg_template:
                    continue
                

                process_cfg_template['write']['cfg']['params']['query'] = {'module': process_cfg_template['write']['cfg']['params']['query']['module']}
                process_cfg_template['write']['cfg']['params']['remove_id']=True
                process_cfg_list = self.client['mongo'].load(**process_cfg_template['write']['cfg']['params'])
                process_cfg_list = list(map(self.resolve_explan_ipfs, process_cfg_list))
                
                cfg['dag'][process_key] =  {
                    'template':process_cfg_template,
                    'clone': process_cfg_list
                }

                

        return cfg

    @staticmethod
    def remove_id(cfg):
        if dict_has(cfg, '_id'):
            print(cfg['_id'], "GOT AN ID")
            del cfg['_id']
        
        return cfg
if __name__ == "__main__":
    import plotly.graph_objects as go
    from commune.utils.misc import dict_fn
    import json


    try:
        with ray.init(address="auto",namespace="commune"):
            process = ConfigManager.deploy(actor=False)
            cfg = process.list_modules()
            # cfg = process.run(module= 'data.regression.crypto.sushiswap.dataset')

            print(cfg)
    except:
        pass
        