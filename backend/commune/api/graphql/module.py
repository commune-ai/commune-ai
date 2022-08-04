import os
import sys
import ray
sys.path.append(os.environ['PWD'])
from commune.utils.misc import dict_put, dict_has, dict_hash
from commune.process import BaseProcess
import inspect

# experiment_manager = ExperimentManager.initialize(spawn_ray_actor=False, actor_name='experiment_manager')





class QueryModule(BaseProcess):
    default_cfg_path = f"{os.environ['PWD']}/commune/api/graphql/manager.yaml"
    cache = {}
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.jobs = {}

    # def ensure_launcher(self):
    #     launcher_loop_running = ray.get(self.launcher.getattr.remote('loop_running'))
    #     if not launcher_loop_running:
    #         self.jobs['launcher.loop'] = self.launcher.loop.remote()

    #     while not launcher_loop_running:
    #         launcher_loop_running = ray.get(self.launcher.getattr.remote('loop_running'))


    def launch(self, job_kwargs, timeout=10, block=True):

        assert 'module' in job_kwargs, 'Module not here bruh'
        assert 'fn' in job_kwargs, 'Function not here fam'

        job_kwargs['kwargs'] = job_kwargs.get('kwargs', {})
        job_kwargs['override'] = job_kwargs.get('override', {'actor.refresh': False } )

        job_id = self.launcher.run_job(**job_kwargs)
        return ray.get(job_id)


    def get_config(self,path, clones=True):
        output_dict = {}
        output_dict['template'] =  self.load_config(cfg=path )
        output_dict['clones'] = ray.get(self.config_manager.find_modules.remote(module=output_dict['template']['module']))
        if not clones:
            return output_dict['template']
        return output_dict

    def get_module_tree(self, root='/app/commune'):
        kwargs = {k:v for k,v in locals().items() if k != 'self'}
        job_hash = inspect.currentframe().f_code.co_name + dict_hash(kwargs)
        print(kwargs, 'KWARGS')
        if job_hash in self.cache:
            return self.cache[job_hash]

        out_dict = {}

        for local_root, dirs, files in os.walk(root, topdown=False):
            for name in files:
                file_path = os.path.join(local_root, name)
                if '.' in file_path and 'yaml' == file_path.split('.')[-1]:
                    
                    cfg = self.config_loader.load(file_path, parse_only=False) 
                    if 'module' in cfg:
                        key_path = file_path.replace(root, '').split('.')[0].replace('/', '.')
                        if key_path[0] == '.':
                            key_path = key_path[1:]
                        dict_put(out_dict,
                            
                                keys=key_path,
                                value= {'config': file_path, 'module': cfg['module']})
        
        self.cache[job_hash] = out_dict
        return out_dict

    @classmethod
    def start(*args,**kwargs):
        with ray.init(address="auto",namespace="commune"):
            return self.deploy(*args,**kwargs)
        

if __name__ == "__main__":
    with ray.init(address="auto",namespace="commune"):
        
        process = QueryModule.deploy(actor=False)
        # print(ray.get(process.launcher.getattr.remote('running_loop')))
        print(process.launch(job_kwargs={
                'module': 'config.manager.ConfigManager', 
                'fn': 'find_modules',
                'kwargs': {'module': 'data.regression.crypto.sushiswap.dataset.Dataset'} ,
                'override': {'actor.refresh': False } ,
                # 'cron': {'name': 'dataset', 'interval': 2}
            }))

        print(process.launch(job_kwargs={
                'module': 'data.regression.crypto.sushiswap.dataset.Dataset', 
                'fn': 'run',
                'kwargs': {} ,
                'override': {'actor.refresh': True } ,
            }))


