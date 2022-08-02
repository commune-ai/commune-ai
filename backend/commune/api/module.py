import os
import sys
import ray
import json
import inspect
sys.path.append(os.environ['PWD'])
from commune.utils.misc import dict_put, dict_has, dict_hash
from commune.process import BaseProcess
from commune.process.launcher.module import Launcher

# experiment_manager = ExperimentManager.initialize(spawn_ray_actor=False, actor_name='experiment_manager')



class APIBase(BaseProcess):
    default_cfg_path = f"api.module"
    def launch(self,module:str='process.bittensor.module.BitModule', fn:str='sync', args:str='[]', kwargs:str='{}', override:dict={} ):
        # print(fn, args, kwargs, 'BRO')
        if isinstance(args, str):
            args = json.loads(args)
        if isinstance(kwargs, str):
            kwargs = json.loads(kwargs)
        
        job_id = ray.get(self.launcher.run_job.remote(module=module, fn=fn, args=args, kwargs=kwargs, override=override))
        return ray.get(job_id)

    def actors(self):
        return ray.get(self.launcher.getattr.remote('actor_names'))

    def get_config(self,path, clones=False):
        output_dict = {}
        output_dict['template'] =  self.load_config(cfg=path )
        output_dict['clones'] = ray.get(self.config_manager.find_modules.remote(module=output_dict['template']['module']))
        if not clones:
            return output_dict['template']
        return output_dict


    def module_tree(self, root='/app/commune'):
        # job_hash = inspect.currentframe().f_code.co_name + dict_hash(kwargs)
        # if job_hash in self.cache:
        #     return self.cache[job_hash]

        out_dict = {}

        for local_root, dirs, files in os.walk(root, topdown=False):
            for name in files:
                file_path = os.path.join(local_root, name)
                if '.' in file_path and 'yaml' == file_path.split('.')[-1]:
                    config_path = file_path
                    module_path = config_path.replace('.yaml', '.py')
                    module_key_path = os.path.dirname(module_path).replace(root, '').lstrip('/').replace('/', '.')
                    
                    dict_put(out_dict,
                                keys=module_key_path,
                                value= {'config': config_path, 'module': module_path})
                   
        # self.cache[job_hash] = out_dict
        return out_dict


if __name__ == "__main__":

    with ray.init(address="auto",namespace="serve"):


        process = APIBase.deploy(actor=False)
        # print(ray.get(process.launcher.getattr.remote('running_loop')))
        # print(process.launch(job_kwargs={
        #         'module': 'process.bittensor.module.BitModule', 
        #         'fn': 'getattr',
        #         'kwargs': {'key': 'current_block'} ,
        #         'override': {'actor.refresh': False } ,
        #         # 'cron': {'name': 'dataset', 'interval': 2}
        #     }))
        print(process.module_tree())
