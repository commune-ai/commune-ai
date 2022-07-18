import warnings
import os, sys
import multiprocessing
sys.path[0] = os.environ['PWD']
from copy import deepcopy
from functools import partial
import re
import ray

import itertools
import torch
from commune.process.base import BaseProcess
from commune.utils.misc import dict_put


os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())
# load from modules


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MODELS = [
    "nbeats.base",
    "deep_ar.base",
    "transformer.base",
    "temporal_fusion_transformer.base",
    "gp.base",
    "ensemble.base",
    "compose.gp_transformer.base",
    "compose.gp_nbeats.base",
    "compose.gp_transformer.distribution",
    "compose.gp_transformer.meta",
    "compose.nbeats_transformer.base",
    "compose.nbeats.base"
]


class Experiment(BaseProcess):

    default_cfg_path = f"{os.getenv('PWD')}/commune/experiment/regression/crypto/module.yaml"
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def start_dataset(self):
        data_manager = self.get_module(self.cfg['data'],actor={'refresh':True})
        ray.get(data_manager.run.remote(tokens=self.cfg['trainer']['tokens'], update=False, run_override_list=['sample_generator']))

    def configure_experiment(self):


        self.cfg['trainer']['data'] = self.cfg['data']
        self.cfg['trainer']['model'] = self.cfg['model']

        if self.cfg.get('debug', {}).get('enable'):
            for k_path,v in self.cfg['debug']['cfg_override'].items():
                dict_put(self.cfg, keys=k_path, value=v)

        self.experiment = self.experiment_manager.get_experiment(name=self.cfg["experiment"]["name"],
                                                                refresh=self.cfg['experiment']['refresh'])


        self.cfg['experiment']['id'] = self.experiment.experiment_id
        self.cfg['trainer']['experiment'] = self.cfg['experiment']
        self.cfg['trainer']['model'] = self.cfg['model']
        self.start_dataset()
        
    def run(self, **kwargs):
        #### RUN THE EXPEIRMENTS FAM ####
        self.configure_experiment()
        if self.cfg['experiment']['num_samples'] > 1:
            self.hyperopt.run(cfg=deepcopy(self.cfg['trainer']), train_job=self.train_job, num_samples=self.cfg['experiment']['num_samples'] )
        else:
            self.train_job(params={},cfg=self.cfg['trainer'])
    
    @staticmethod
    def train_job(params={}, cfg=None):
        # update arguments with config
        trainer = BaseProcess.get_object(cfg['module'])(cfg)
        trainer.run(params=params)  # train the experiment

# type determines the type of the argument

if __name__ == "__main__":
    with ray.init(address="auto",namespace="commune"):
        Experiment.deploy(actor=False).run()