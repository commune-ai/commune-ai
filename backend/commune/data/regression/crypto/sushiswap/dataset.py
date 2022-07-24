
from copy import deepcopy
import sys
import datetime
import os
sys.path.append(os.environ['PWD'])
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
                        dict_put,dict_get
                        )

from commune.transformation.block.hash import String2IntHash
from commune.ray import ActorBase
from commune.process import BaseProcess


class Dataset(BaseProcess):
    default_cfg_path = f"{os.getenv('PWD')}/commune/data/regression/crypto/sushiswap/dataset.yaml"

    def __init__(self,cfg, client=None):

        super().__init__(cfg=cfg)

        self.init_maps()
        self.max_actor_count = cfg['actor_count']
        self.round_robin_actor_id = -1
        self.queue = {}
        self.generators_running = {'train': False, 'val': False}
        self.pipeline_override()

    def change_state(self):
        self.cfg['write']['cfg']['params']['query']['name'] = self.cfg['module']
        print("WRITING BRUH ", self.cfg['write'])


    def get_batch(self, timestamps=[],timescale='30m', device='cuda:0'):
        job_list = []
        for actor in self.actors:
            job_list.append(actor.get_batch.remote(timestamps=timestamps, timescale=timescale))
        
        out_batch_list = ray.get(job_list)
        out_batch = torch_batchdictlist2dict(out_batch_list, dim=0)
        out_batch = self.convert_device(out_batch,device=device)

        return out_batch
    def get_split_batch(self,split='train', device='cuda:0', periods=None):

        batch = self.queue[split].get(True)
        batch = self.convert_device(batch,device=device)
        batch = ray.get(self.actors[0].resize_batch_periods.remote(batch=batch, periods=periods))

        return batch
    def stop_generators(self, splits=['val', 'train']):
        job_list = []
        for split in splits:
            for actor in self.actors:
                job_list.append(actor.stop_generators.remote(split=split))
            self.generators_running[split] = False
        ray.get([*job_list, *self.generator_jobs])

    
    # def generators_running(self, split='train'):
    #     return any(ray.get([a.is_running.remote(split) for a in self.actors]))
    
    def start_generators(self,
                        splits=['train','val'], 
                        restart=True,
                        timescales=['15m'], 
                        skip_step=5,
                        batch_size=32,
                        queue_limit=40):

        
        self.generator_jobs = []

        for split in splits:
           
            if self.generators_running[split] and not restart:
                continue 

            self.queue[split] = Queue(queue_limit)
            for actor in self.actors:
                self.generator_jobs.append(actor.start_generators.remote(split=split, 
                                                        restart=restart, 
                                                        queue=self.queue[split],
                                                        timescales=timescales, 
                                                        skip_step=skip_step,
                                                        batch_size=batch_size))
                
            self.generators_running[split]= True


        return None


    def spawn_actor(self,
                     cfg
                     ):
        actor_name = f"{cfg['actor']['name']}.{self.round_robin_actor_id}" 
        actor_handle = self.get_module(cfg, actor= {'refresh': True,'name': actor_name})      
        self.map['actor_name']['actor_handle'][actor_name] = actor_handle

        return actor_handle, actor_name 

        


    def get_actor(self):        
        """
        This facilitates round robin allocation of tokens to their respective actor placeholders
        """

        self.round_robin_actor_id += 1
        self.round_robin_actor_id = self.round_robin_actor_id%self.max_actor_count

        if self.max_actor_count > len(self.actors):
            self.spawn_actor(cfg=deepcopy(self.cfg['pipeline']))

        actor_handle = self.actors[self.round_robin_actor_id]
        actor_name =  f"{self.cfg['pipeline']['actor']['name']}.{self.round_robin_actor_id}"

        return actor_handle, actor_name

    def process(self, **kwargs):
        # self.kill_actors()


        default_add_tokens_kwargs = dict(
                  tokens=[],
                  verbose=True,
                  run_override_list=None,
                  refresh=True,
                  update=False
        )
        add_token_kwargs = {k:kwargs.get(k,v ) 
                            for k,v in default_add_tokens_kwargs.items()}
        self.add_tokens(**add_token_kwargs)



        default_start_generators_kwargs = dict(
                                            splits=list(self.cfg['pipeline']['splits'].keys()), 
                                            batch_size=self.cfg['pipeline']['batch_size'],
                                            timescales=self.cfg['timescales'], 
                                            skip_step=self.cfg['pipeline']['skip_step'],
                                            restart=True
        )
        start_generators_kwargs = {k:kwargs.get(k,v ) 
                                for k,v in default_start_generators_kwargs.items()}

        self.start_generators(**start_generators_kwargs)

    def add_tokens(self,
                  tokens=[],
                  verbose=True,
                  run_override_list=None,
                  refresh=True,
                  update=False):

        if len(tokens) == 0:
            tokens = self.cfg['tokens']

        running_job_list = []
        finished_job_list = []

        for token in tokens:
            if (token in self.map['token']['actor_name']):
                actor_name = self.map['token']['actor_name'][token]
                actor_handle = self.map['actor_name']['actor_handle'][actor_name]
            else:    
                # round robin allocation of tokens to data actors
                actor_handle, actor_name = self.get_actor()   

            job_id = actor_handle.run_dag.remote(tokens=[token],
                                                          run_override_list=run_override_list,
                                                          update=update
                                                          )       
            running_job_list.append(job_id)     
            self.map['token']['actor_name'][token] = actor_name


            if actor_name in self.map['actor_name']['tokens']:
                self.map['actor_name']['tokens'][actor_name].append(token)
            else:
                self.map['actor_name']['tokens'][actor_name] = [token]

            if verbose:
                print(f"Allocating {token} to ({actor_name},{actor_handle})")
                print(f"Finished: {len(finished_job_list)}"
                    f"Running: {len(running_job_list)}"
                    f" Total Jobs: {len(tokens)}")


        finished_job_list = []

        while running_job_list:
            tmp_finished_job_list, running_job_list = ray.wait(running_job_list)
            finished_job_list = list(set([*finished_job_list,*tmp_finished_job_list]))
        
        ray.get(finished_job_list)

    def get_info(self, actor_index=0):
        return ray.get(self.actors[actor_index].get_info.remote())

    @property
    def actor_names(self):
        return list(self.map['actor_name']['actor_handle'].keys())

    @property
    def default_actor_names(self):
        actor_name_list = []
        for i in range(self.max_actor_count):
            actor_name_list += [f"{self.cfg['pipeline']['actor']['name']}.{i}"]
        
        return actor_name_list


    @property
    def actors(self):
        return list(self.map['actor_name']['actor_handle'].values())

    @property
    def actor_count(self):
        return len(self.actors)
    @property
    def token2actor(self):
        return [self.map['actor_name']['actor_handle'] for v in self.map['token']['actor_name'].values()]

    def actor2token(self):
        return self.map['actor_name']['tokens']

    @property
    def tokens(self):
        return list(self.map['token']['actor_name'].keys())

    def get_actors_by_token(self, tokens=None):
        if tokens:
            return [self.map['actor_name']['actor_handle'][token] for token in tokens]
        else:
            return self.actors

    def get_pipeline_map(self):
        self.pipeline_map = {}
        running_job_list = [actor.get_pipeline_map.remote() for actor in self.actors]
        while running_job_list:
            finished_job_list,running_job_list = ray.wait(running_job_list) 
            if len(finished_job_list)>0:
                for pipeline_map in ray.get(finished_job_list):
                    self.pipeline_map.update(pipeline_map)

        # print(self.pipeline_map.keys())
        return self.pipeline_map
    def num_batches(self):
        num_batches = 0
        for name, actor in zip(self.actor_names,self.actors):
            num_batches += ray.get(actor.get.remote('num_batches'))
            
    def remove_actor(self, actor):
        self.kill_actor(actor)


        for m in ['actor_handle', 'tokens']:
            if actor in self.map['actor_name'][m]:
                del self.map['actor_name'][m][actor]

        for m in ['token']:
            for k  in deepcopy(list(self.map[m]['actor_name'].keys())):
                print(self.map[m]['actor_name'][k], k)
                if actor == self.map[m]['actor_name'][k]:
                    del self.map[m]['actor_name'][k]   
        
        # # # update the maps
        # for to_key in self.map['actor_handle'].keys():
        #     del self.map['actor_handle'][to_key][actor]

    def remove_actors(self):
        for actor in self.default_actor_names:
            self.remove_actor(actor)

        self.round_robin_actor_id = -1
        self.init_maps() 

    def restart_actors(self):
        job_list = []
        for actor_name in self.default_actor_names:
            if self.actor_exists(actor_name):
                actor = ray.get_actor(actor_name)
                ray.get(actor.restart.remote())


    def init_maps(self):
        # token to actor map and reverse
        self.map = {}

        # map actor to multiple token pairs (1 to many)
        dict_put(self.map,['actor_name','actor_handle'], {})
        dict_put(self.map,['token','actor_name'],{})
        dict_put(self.map,['actor_name','tokens'],{})





    def stratify_timescale_token_batch(self, batch_dict,
                            token_hash_tensor: torch.Tensor,
                            timescale_hash_tensor: torch.Tensor):

        # get token to hash
        unique_token_hash = torch.unique(token_hash_tensor).tolist()
        
        token2hash_dict = dict(zip(map(lambda x: String2IntHash.inverse(x), unique_token_hash), unique_token_hash))
        # get timescale to hash
        unique_timescale_hash = torch.unique(timescale_hash_tensor).tolist()
        timescale2hash_dict = dict(zip(map(lambda x: String2IntHash.inverse(x), unique_timescale_hash), unique_timescale_hash))

        timescale_token_batch_dict = {}
        for timescale_key, timescale_hash in timescale2hash_dict.items():
            timescale_token_batch_dict[timescale_key] = {}
            for token_key, token_hash in token2hash_dict.items():

                timescale_token_batch_idx = ((token_hash == token_hash_tensor)*(timescale_hash==timescale_hash_tensor)).nonzero().squeeze(1)
                timescale_token_batch_dict[timescale_key][token_key] = {k:v[timescale_token_batch_idx]
                                                                     for k,v in batch_dict.items() if isinstance(v,torch.Tensor)}

        return timescale_token_batch_dict
    
    @staticmethod
    def combine_decentralized_batch( token_batch_jobs):
        out_batch_list = ray.get(token_batch_jobs)
        out_batch = torch_batchdictlist2dict(out_batch_list, dim=0)
        return out_batch

    @property
    def token(self):
        return self.cfg['tokens']

    def convert_device(self,x, try_count_limit = 10, device='cuda:0'):
        
        for k,v in x.items():
            try_count = 0
            while try_count < try_count_limit:
                try:
                    x[k] = v.to(device)
                    break
                except RuntimeError:
                    try_count += 1
        return x

    def pipeline_override(self):
        if 'pipeline' in self.cfg:
            for process_key, process_cfg in self.cfg['pipeline_override'].items():
                self.cfg['pipeline']['dag'][process_key]['refresh'] = bool(process_cfg.get('refresh'))
                self.cfg['pipeline']['dag'][process_key]['run'] = bool(process_cfg.get('run'))
                print(process_cfg)



if __name__=="__main__":

    with ray.init(address="auto", namespace='commune'):
        dataset = Dataset.deploy(actor={'refresh': True})
        ray.get(dataset.run.remote(update=False))

        # ray.get(dataset.run.remote(update=False))

    