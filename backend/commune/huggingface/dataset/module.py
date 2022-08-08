
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import transformers
from copy import deepcopy
import 
from typing import Union

from commune.process.base import BaseProcess
import torch
import ray
from commune.utils.misc import dict_put

class DatasetModule(BaseProcess):
    default_cfg_path= 'commune.huggingface.dataset.module'

    def __init__(self, cfg=None):
        BaseProcess.__init__(self, cfg=Module)
        self.load()
        self.pipeline

    def load(self):
        self.dataset = self.load_dataset()


    def load_dataset(self, *args, **kwargs):

        if len(args) == 0 and len(kwargs) == 0:
            args = self.cfg['dataset']
            kwargs = {}

            if type(args) == str):
                args = [args]
            if type(args) == dict:
                kwargs = args
                args = []
            
        return datasets.load_dataset(*args, **kwargs)


        
        if isinstance(dataset_kwargs, str):
            dataset_args = []

        if 
        datasets.load_dataset(**)
    def load_tokenizer(self, *args, **kwargs):
        tokenizer_kwargs = 

    def save(self,dataset=None, *args, **kwargs):

        if dataset != None:
            return dataset.save_to_disk(*args, **kwargs)

    def push_to_hub(self,dataset=None, **args, **kwargs):
        
        if dataset != None:
            return dataset.push_to_hub(*args, **kwargs)



if __name__ == '__main__':
    with ray.init(address="auto",namespace="commune"):
        model = ModelModule.deploy(actor={'refresh': False})
        sentences = ['ray.get(model.encode.remote(sentences))', 'ray.get(model.encoder.remote(sentences)) # whadup fam']
        print(ray.get(model.self_similarity.remote(sentences)))

        