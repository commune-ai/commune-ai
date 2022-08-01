
from copy import deepcopy
import sys
import datetime
import os
import asyncio
import multiprocessing
import torch
sys.path.append(os.environ['PWD'])
from commune.config import ConfigLoader
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
                        nan_check,
                        dict_put, dict_has, dict_get, dict_hash
                        )

from commune.transformation.block.hash import String2IntHash
from commune.ray import ActorBase
from commune.process import BaseProcess



class Launcher(BaseProcess):

    default_cfg_path = f"process.launcher.module"
    def setup(self):
        self.actor_map = {}
        self.actor2jobs = {}
        self.cron_jobs = {}

        self.job2queue = {}
        self.cfg['queue'] = {
            'in': 'launcher.in',
            'out': 'launcher.out'
        }


        # self.spawn_actors()
    @property
    def max_actor_count(self):
        return self.cfg['max_actor_count']

    def get_job_results(job_kwargs):
        dict_hash(job_kwargs)

    def send_job(self, job_kwargs, block=False):
        print('BRO')
        self.client['ray'].queue.put(topic=self.cfg['queue']['in'], item=job_kwargs, block=block )
        

    def run_job(self, module, fn, kwargs={}, args=[], override={}, cron=None):

       
        actor, actor_name = self.launch_actor(module=module,override=override)

        job_id = getattr(actor, fn).remote(*args,**kwargs)
        job_kwargs = {'actor_name':actor_name,
                        'fn': fn,
                        'kwargs': kwargs}

        self.register_job(actor_name=actor_name, job_id=job_id)
        # if cron:
        #     self.register_cron(name=cron['name'], interval=cron['interval'], 
        #                         job = job_kwargs )


        self.client['ray'].queue.put(topic=self.cfg['queue']['out'],item=job_id)


        return job_id

    @property
    def resource_limit(self):
        return {'gpu': torch.cuda.device_count(), 
                'cpu': multiprocessing.cpu_count()}

    def register_job(self, actor_name, job_id):
        if dict_has(self.actor2jobs,[actor_name,'running'] ):
            self.actor2jobs[actor_name]['running'].append(job_id)
        else:
            dict_put(self.actor2jobs, [actor_name,'running'], [job_id])

    def get_jobs(self, mode='running', actors = None):

        if actors is None:
            actors = list(self.actor2jobs)
        self.update_actor2jobs(actors=actors)

        job_list= []
        for actor in actors:
            actor_jobs =  self.actor2jobs[actor]
            job_list.extend(actor_jobs[mode])

        return job_list


    def update_actor2jobs(self, actors=None):
        if actors == None:
            actors = list(self.actor2jobs.keys())


        for actor in actors:
            actor_jobs = self.actor2jobs[actor]
            if 'running' in actor_jobs:
                actor_jobs['finished'], actor_jobs['running'] = ray.wait(actor_jobs['running'])
                self.actor2jobs[actor] = actor_jobs

                # for job in actor_jobs['finished']:
                #     queue_topic = self.job2queue.get(job.hex(), self.cfg['queue']['in'])
                #     out_item = ray.get(job)
                #     self.client['ray'].queue.put(topic=queue_topic,item=ray.get(job))


    def load_balance(self, proposed_actor = None):

        while self.actor_count >= self.max_actor_count:
            for actor_name in self.actor_names:
                running_actor_jobs = self.get_jobs(actors=[actor_name], mode='running')
                if len(running_actor_jobs) == 0 and \
                        proposed_actor!=actor_name:

                    self.remove_actor(actor_name)
            
                

    def launch_actor(self, module, override={} , **kwargs):
        # self.load_balance(proposed_actor=actor_name)
        actor = self.get_module(cfg=module, actor=True, override=override) 
        
        actor_name = self.register_actor(actor=actor)
        return actor, actor_name

    def register_actor(self, actor):
        actor_name = ray.get(actor.getattr.remote('actor_name'))
        self.actor_map[actor_name] = actor
        return  actor_name

    def process(self, **kwargs):

        run_job_kwargs = self.client['ray'].queue.get(topic=self.cfg['queue']['in'], block=True )        
        # print(run_job_kwargs,'BRO')
        self.run_job(**run_job_kwargs)
        out_item = ray.get(self.get_jobs('finished'))


    @property
    def actor_names(self):
        return list(self.actor_map.keys())

    @property
    def actors(self):
        return list(self.actor_map.values())

    @property
    def actor_count(self):
        return len(self.actors)

    def remove_actor(self,actor):
        '''
        params:
            actor: name of actor or handle
        '''

        assert actor in self.actor_map, 'Please specify an actor in the actor map'
        self.kill_actor(actor)
        
        del self.actor2jobs[actor]
        del self.actor_map[actor]

  

    def remove_all_actors(self):
        for actor in self.actor_names:
            self.remove_actor(actor)

    @staticmethod
    def getArgparse():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', dest='mode', default='server')
        input_args = parser.parse_args()
        return input_args




if __name__=="__main__":

    with ray.init(address='auto', namespace='commune'):

        input_args = Launcher.getArgparse()

        # print(ctx)


        mode = input_args.mode

        if mode == 'server': 
            # print(Launcher.get_actor('launcher'))
            launcher = Launcher.deploy(actor={'refresh': True})
            print('bro')
            ray.get(launcher.loop.remote(verbose=False))
        elif mode == 'client':
            launcher = Launcher.deploy(actor={'refresh': False})
            client = Launcher.default_clients()
            
            job_list = [launcher.send_job.remote(job_kwargs={
                'module': 'process.launcher.module.Launcher', 
                'fn': 'get_jobs',
                'kwargs': {} ,
                # 'override': {'actor.refresh': True }
                # 'cron': {'name': 'dataset', 'interval': 2}
            }, block=True)]

            # print(launcher.getattr.remote('cfg.queue'))

            # print(ray.get(launcher.getattr.remote('cfg')))
            print(client['ray'].queue.get('launcher.out'))
# 

                    

        


