import os
import sys
import ray
sys.path.append(os.environ['PWD'])
from commune.utils.misc import dict_put, dict_has, dict_hash
from commune.process import BaseProcess


# experiment_manager = ExperimentManager.initialize(spawn_ray_actor=False, actor_name='experiment_manager')



class QueryModule(BaseProcess):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/graphql/manager.yaml"
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.jobs = {}

    def ensure_launcher(self):
        launcher_loop_running = ray.get(self.module['launcher'].getattr.remote('loop_running'))
        if not launcher_loop_running:
            self.jobs['launcher.loop'] = self.module['launcher'].loop.remote()

        while not launcher_loop_running:
            launcher_loop_running = ray.get(self.module['launcher'].getattr.remote('loop_running'))


    def launch(self, job_kwargs, block=True):

        assert 'module' in job_kwargs
        assert 'fn' in job_kwargs
        job_kwargs['kwargs'] = job_kwargs.get('kwargs',{})
        job_kwargs['override'] = job_kwargs.get('override', {'actor.refresh': False } )
        job_kwargs['cron'] = job_kwargs.get('cron', None) #'cron': {'name': 'dataset', 'interval': 2}

        self.ensure_launcher()

        print('BRUHHHHH')

        ray.get(self.module['launcher'].send_job.remote(job_kwargs=job_kwargs, block=True))
        
        
        return self.client['ray'].queue.get(topic=job_kwargs['queue'], 
                                            block=block)


if __name__ == "__main__":

    with ray.init(address="auto",namespace="commune"):


        process = QueryModule.deploy(actor=False)
        # print(ray.get(process.module['launcher'].getattr.remote('running_loop')))
        print(process.launch(job_kwargs={
                'module': 'config.manager.ConfigManager', 
                'fn': 'run',
                'kwargs': {'query':{'module': 'data.regression.crypto.sushiswap.dataset'}} ,
                'override': {'actor.refresh': False } ,
                # 'cron': {'name': 'dataset', 'interval': 2}
            }).keys())


