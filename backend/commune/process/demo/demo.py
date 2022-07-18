
from copy import deepcopy
import sys
import datetime
import os
import time
import logging
sys.path.append(os.environ['PWD'])
from commune.config import ConfigLoader
from commune.ray.utils import kill_actor, create_actor
import ray
from ray.util.queue import Queue
import torch
from commune.utils.ml import tensor_dict_check
from commune.utils.misc import  (RunningMean,
                        chunk,
                        get_object,
                        even_number_split,
                        torch_batchdictlist2dict,
                        round_sig,
                        timer,
                        tensor_dict_shape,
                        nan_check
                        )

from commune.transformation.block.hash import String2IntHash
from commune.ray import ActorBase
from commune.process import BaseProcess

class Demo(BaseProcess):
    default_cfg_path="process.base.demo"
    def __init__(self, cfg):
        super().__init__(cfg)
        print("FUCK")
        self.counter = 0

    def process(self, **kwargs):
        self.counter+= 1
        self.object_dict['count'] = self.counter
        print(self.actor_name,self.object_dict)
        return "BROOOOOOOOOOOO"

if __name__ == '__main__':
    
    with ray.init(address="auto",namespace="commune3"):
        Demo.kill_actor('queue_server')

        modules = [Demo.deploy(actor={'name': f'demo{i}', 'refresh': True}) for i in range(15)]

        client = BaseProcess.default_client()

        [client['ray'].queue.delete_topic(f'fam{i}') for i in range(len(modules))]
        # [client['ray'].queue.create_topic(f'fam{i}') for i in range(len(modules))]
        
        jobs = [
            modules[0].run.remote(override={'cfg.queue.out': f'fam{0}'}),
            *[m.run.remote(override={'cfg.queue.in': f'fam{i}', 'cfg.queue.out': f'fam{i+1}'}) for i,m in enumerate(modules[1:])]
        ]


        ray.get(jobs[0])
        print(ray.get(client['ray'].queue.server.getattr.remote('queue')))

        while jobs:
            fin , jobs = ray.wait(jobs)
            ray.get(fin)
            print(len(jobs))
        
        




        time.sleep(0.5)
        

        # print(Demo.kill_actor('demo1'))
        # print(ray.get_actor('demo1'))
        # print(ray.get(module[1].get.remote('object_dict')))


        