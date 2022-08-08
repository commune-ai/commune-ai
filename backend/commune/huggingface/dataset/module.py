
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
        # loads the dataset
        self.dataset = self.load_dataset()

        # loads the pipeline
        self.pipeline = self.load_pipeline()


    def load_dataset(self, *args, **kwargs):

        if len(args) + len(kwargs) == 0:
            args = self.cfg.get('dataset')
            assert type(args) in [str, dict, list]

            kwargs = {}

            if type(args) == str):
                args = [args]
            if type(args) == dict:
                kwargs = args
                args = []
            if type(args) == list:
                args = args
                kwargs = {}
            
        return datasets.load_dataset(*args, **kwargs)


        
        if isinstance(dataset_kwargs, str):
            dataset_args = []

        if 
        datasets.load_dataset(**)
    def load_pipeline(self, *args, **kwargs):
        
        if len(args) + len(kwargs) == 0:
            kwargs = self.cfg.get('pipeline')
            assert type(kwargs) != None 
            if type(kwargs)  == str:
                transformer.AutoTokenizer.from_pretrained(kwargs) 
            else:
        else:
            raise NotImplementedError
        if pipeline_kwargs == None

    def save(self,dataset=None, *args, **kwargs):

        if dataset != None:
            return dataset.save_to_disk(*args, **kwargs)

    def push_to_hub(self,dataset=None, **args, **kwargs):
        
        if dataset != None:
            return dataset.push_to_hub(*args, **kwargs)

    def load_pipeline(self):
        pipeline_cfg = self.cfg.get('pipeline')
        assert pipeline_cfg != None

        if isinstance(pipeline_cfg, list):
            pipeline = range(len(pipeline_cfg))
        elif isinstance(pipeline_cfg, dict):
            pipeline = list(pipeline_cfg.keys())

        for process_key in pipeline_keys:

            process = self.pipeline[process_key]
            process_cfg = pipeline_cfg[process_key]

            process_params = process_cfg.get('params', {})
            process_input_keys = process_cfg.get('input')
            process_output_keys = process_cfg.get('output')

            if isinstance(process_input_keys, str):
                process_input_keys = [process_input_keys]
            if isinstance(process_output_keys, str):
                process_output_keys = [process_poutput_keys]

            def process_fn(inputs:dict):
                input_kwargs, input_args = {}, []
                
                inputs_type = type(inputs)
                input_keys_type = type(input_keys)

                if input_keys_type in [list]:
                    input_args = [inputs[k] for k in input_keys]
                elif input_keys_type in [dict]:
                    input_kwargs = {k:inputs[v] for k,v in input_keys.items()}
                elif input_keys = None:
                    input_args = [inputs]

                outputs = process(*input_arg, **input_kwargs, **process_params)

                '''
                map outputs back to input sample
                '''
                outputs_type = type(outputs) 
                output_keys_type = type(output_keys)
                
                assert len(outputs) == len(output_keys)

                if outputs_type in [list, tuple]:
                    if output_keys_type == dict:
                        output_keys = list(output_keys.values())

                    for i in range(len(output_keys)):
                        inputs[output_keys[i]] = outputs[i]
                    
                elif outputs_type in [dict]:
                    if output_keys_type == dict:
                        outputs = {k:outputs[v] for k,v in output_keys.items()}
                    elif output_keys_type == list:
                        outputs = {k: outputs[k] for k in output_keys}
                
                    inputs.update(outputs)
                elif output_keys == None:
                    return outputs

                return inputs
        
            dag.append(process_fn)

        return dag


if __name__ == '__main__':
    with ray.init(address="auto",namespace="commune"):
        model = ModelModule.deploy(actor={'refresh': False})
        sentences = ['ray.get(model.encode.remote(sentences))', 'ray.get(model.encoder.remote(sentences)) # whadup fam']
        print(ray.get(model.self_similarity.remote(sentences)))

        