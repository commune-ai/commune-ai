
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import transformers
import 
from typing import Union

from commune.process.base import BaseProcess
import torch
import ray
from commune.utils.misc import dict_put

class ProcessModule(BaseProcess):
    default_cfg_path= 'commune.huggingface.process.module'

    def __init__(self, cfg=None):
        BaseProcess.__init__(self, cfg=cfg)
        self.process = self.load_process()
        self.dag = self.compose_dag()
    

    def compose_dag(self):
        dag = []  
        dag_cfg = self.cfg.get('dag', None)

        if isinstance(self.dag, list):
            dag_keys = range(len(self.dag))
        elif isinstance(self.dag, dict):
            dag_keys = list(self.dag.keys())

        for dag_key in dag_keys:

            process = self.process[dag_key]
            process_cfg = dag_cfg[dag_key]
            process_params = process_cfg.get('params', {})
            input_keys = process_cfg.get('input', None)
            output_keys = process_cfg.get('output', input_keys)

            if isinstance(input_keys, str):
                input_keys = [input_keys]
            if isinstance(output_keys, str):
                output_keys = [output_keys]

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

    def run(self, example):
        for process in dag:
            example = process(example)
        
        return example

        

    def load(self):
        process_dict = {}
        for p_key, p_kwargs in self.cfg['load'].items():
            mode = p_kwargs.pop('mode')
            if mode == 'tokenizer':
                process_module = transformer.AutoTokenizer.from_pretrained(**p_kwargs)
            else:
                raise NotImplementedError
    
            process_dict[p_key] =  process_module

        return process_dict


        
       

    def push_to_hub(self,dataset=None, **args, **kwargs):
        
        if dataset != None:
            return dataset.push_to_hub(*args, **kwargs)



if __name__ == '__main__':
    
        