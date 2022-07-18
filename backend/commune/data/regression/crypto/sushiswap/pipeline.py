
import itertools
from torch.utils.data import Dataset, DataLoader
from commune.utils.misc import get_object
from filelock import FileLock
import random
from commune.transformation.block.hash import String2IntHash
import torch
import ray
from ray.util.multiprocessing import Pool
from ray.util.queue import Queue
import os
import datetime
from copy import deepcopy
import gc
import numpy as np
from commune.ray.utils import kill_actor, create_actor
from commune.utils.misc import chunk, get_object, dict_get, torch_batchdictlist2dict
from commune.ray.actor import ActorBase
from commune.process import BaseProcess

class Pipeline(BaseProcess):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/data/regression/crypto/sushiswap/pipeline.yaml"
    def __init__(
            self,
            cfg
    ):
        super().__init__(cfg=cfg)
        self.sample_adapter = {}
        self.generators_running = {split:False for split in self.cfg['splits']}
        self.tokens = []
        self.dag = self.build_dag()
        self.last_update_timestamp = {}
        self.last_update_delay = {}

    def build_dag(self):
        '''
        Build the MotherFuckin DAG
        '''

        dag = {}
        
        # get a map of the client connections

        for proc_key, proc_cfg in self.cfg['dag'].items():
            proc_class = get_object('commune.' + proc_cfg['module'])
            proc_cfg = deepcopy(proc_cfg)
            proc_cfg['client'] = self.client_manager
            dag[proc_key] = proc_class(cfg=proc_cfg)

        self.dag = dag
        return dag

    def run_token_dag(self, token,
                     run_override_list = [],
                     update=False):

        dag = self.build_dag()

        dag_step_name_list = deepcopy(list(dag.keys()))

        override_dict = {"cfg.token": token,
                       "cfg.base_ticker": deepcopy(self.cfg["base_ticker"]),
                       "cfg.periods": deepcopy(self.cfg["periods"]),
                       "cfg.feature_group": deepcopy(self.cfg["feature_group"]),
                       "cfg.known_future_features": deepcopy(self.cfg["known_future_features"])}

        
        
        last_step_name = dag_step_name_list[-1]


        for dag_step_name in dag_step_name_list:
            dag_step = dag[dag_step_name]
            if run_override_list:
                dag_step.cfg['run'] = bool(dag_step_name in run_override_list)

            if dag_step.cfg['run']:
                dag_step.run(override=override_dict)
        
            if last_step_name == dag_step_name:
                self.sample_adapter[token] = dag_step
    

        
        self.last_update_delay[token] =   (datetime.datetime.utcnow().timestamp() - self.last_update_timestamp.get(token, 0))//60
        self.last_update_timestamp[token] =  datetime.datetime.utcnow().timestamp()

    
    def get_pipeline_list(self):
        return list(self.cfg['dag'].keys())

    def get_info(self, token=None):
        if token is None:
            token = self.tokens[0]
        return self.sample_adapter[token].get_info()

    def process(self, **kwargs):
        print("PIPELINE PROCESS")
        self.run_dag(**kwargs)

    def run_dag(self, 
                tokens=None,
                run_override_list=None,
                update=False):
        '''
        Run the MotherFucking DAG
        '''
        for token in tokens:
            
            # for each token run the timescales,
            self.run_token_dag(token=token,
                                      run_override_list=run_override_list,
                                      update=update)

            self.tokens.append(token)
            self.tokens = list(set(self.tokens))

        

    def stop_generators(self, split='train'):
        self.generators_running[split] = False

    @property
    def split_keys(self):
        return self.sample_adapter.split_keys

    def is_running(self,split='train'):
        return self.generators_running[split]
    

    def start_generators(self,
                              queue=Queue(maxsize=100),
                              restart=False,
                              split="train", 
                              timescales=['15m'], 
                              skip_step=10, 
                              batch_size=32):
        
        self.generators_running[split] = True
        batch_generators = {}
        for token in self.tokens:
            batch_generators[token] = self.sample_adapter[token].batch_generator(split=split,
                                                                               timescales=timescales, 
                                                                               skip_step=skip_step,
                                                                               batch_size=batch_size)               
        finish_datasets = []
        cnt = 0 
        while  self.generators_running[split]:
            for token in self.tokens:
                if token not in finish_datasets:
                    batch = next(batch_generators[token])
                    
                    if batch == None: 
                        if restart:
                            batch_generators[token] = self.sample_adapter[token].batch_generator(split=split,
                                                                               timescales=timescales, 
                                                                               skip_step=skip_step,
                                                                               batch_size=batch_size)
                        else:
                            finish_datasets.append(token)
                    else: 
                        queue.put(batch)


                if len(finish_datasets) == len(self.tokens):
                    self.generators_running[split] = False
                    break


    def resize_batch_periods(self, batch, periods=None, token=None):
        if periods is None:
            periods = self.cfg['periods']
        if token is None:
            token = self.tokens[0]

        generated_periods = self.cfg['generated_periods']

        input_bounds = [generated_periods['input']-periods['input'], generated_periods['input']]
        output_bounds = [generated_periods['input'], generated_periods['input']+periods['output']]

        for input_key in self.sample_adapter[token].input_columns + self.sample_adapter[token].meta_keys:
            batch[input_key] = batch[input_key][:, input_bounds[0]:output_bounds[1]]

        for gt_key in self.cfg['gt_keys']:
            for suffix in ['', '-raw']:
                batch[f"gt_past_{gt_key}{suffix}"] = batch[f"gt_past_{gt_key}{suffix}"][:, -periods['input']: ]
                batch[f"gt_future_{gt_key}{suffix}"] = batch[f"gt_future_{gt_key}{suffix}"][:, :periods['output']]

        return batch

    def get_batch(self, timestamps, timescale, periods=None):
        if periods is None:
            periods = self.cfg['periods']
        batch_list = []
        
        for token in self.tokens:
            batch = self.sample_adapter[token].get_batch(timestamps=timestamps, timescale=timescale)
            batch = self.resize_batch_periods(batch=batch, periods=periods)
            batch_list.append(batch)

        batch_list = torch_batchdictlist2dict(batch_list)

        return batch_list

    def get_pipeline_map(self):
        pipeline_map =  {token: self.sample_adapter[token].get_pipeline_map() for token in self.tokens}
        return pipeline_map
        
    def should_update(self, token):
        delay  = self.sample_adapter[token].get_delay()
        return delay > self.cfg['delay_threshold']
