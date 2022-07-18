

import os
import random
import datetime
import torch
import numpy as np
from copy import deepcopy
import time
import re
import math
import itertools
from commune.transformation.block.hash import String2IntHash
import random
import torch
import mlflow
from mlflow.tracking import MlflowClient
import ray
from ray import tune
from commune.model.utils import scale_model_dropout
from commune.experiment.utils import  log_any_artifact
from commune.utils.misc import  (RunningMean,
                        get_object,
                        even_number_split,
                        torch_batchdictlist2dict,
                        round_sig,
                        timer,
                        tensor_dict_shape,
                        nan_check,
                        dict_fn,
                        dict_put,
                        dict_has,
                        dict_get

                        )

from commune.config.utils import resolve_devices
from commune.process.base import BaseProcess




# metric functions
# loss functions
class RegressionOracleTrainer(BaseProcess):

    default_cfg_path = f"{os.getenv('PWD')}/commune/trainer/regression/crypto/trainer.yaml"
    def __init__(self, cfg: dict):
        super().__init__(cfg=cfg)
        
        cfg = resolve_devices(cfg, device = cfg['device'])
        torch.cuda.empty_cache()
        # we want to add higher layer for the token being trained on


        self.cfg = cfg

        # get the tranformation function and include it into the data
        """
        Get Dataset Actor Manager
        """
        self.data = self.get_module(self.cfg['data'], actor={'refresh': False})
        """
        Initialize the train state and teh exeprimen logging
        """
  

        """
        Get the Models that connect to the data
        """
        self.init_experiment()
        self.init_train_state()
        # initialize the model
        self.model = self.get_module(cfg['model'], override={'sub_module.data':self.cfg['data']})
        self.model.setattr('cfg.experiment.run.artifact_uri', self.run_obj.info.artifact_uri)
        self.model.setattr('cfg.experiment.run.id', self.run_id)
        # load the checkpoint

    def init_experiment(self):
        """
        Initialize experiment logging

        - Initialize the manager
        - set up experiment tags (append tags if they already exist (adding same experiment but with different tokens))
        :return:
        """
        cfg = self.cfg

        """
        if the tags exist already, we need to append them, especially if they are lists
        - ie if you are adding a run with a new set of tokens, or another type of model
        """
        self.experiment = self.experiment_manager.get_experiment(cfg['experiment']['name'])
        self.run_obj = self.experiment_manager.create_run(experiment_id = cfg['experiment']['id'])
        self.run_id = self.run_obj.info.run_id

        self.cfg['experiment']['run_id'] = self.run_id

        run_tags = {"model":  cfg['model']['module'],
                    "data": cfg['data']['module'],
                    "base_ticker":cfg['data']['pipeline']['base_ticker'], # get the name of one data
                    "tokens": cfg['tokens'],
                    "timescales": cfg['timescales'],
                               }


        self.experiment_manager.set_tags(self.run_id, run_tags)




    def init_train_state(self):
        cfg = self.cfg
        self.train_state = {
            "current_time": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "es_step": 0,
            "stop_early": False,

            "reduce_lr_step": 0,
            "reduce_lr_factor": cfg['reduce_lr_factor'],
            "reduce_lr_criteria": cfg['reduce_lr_criteria'],
            "samples_per_second": {},
            "epoch_index": 0,
            "batch_index": 0,

            "score_dict": {"train": [], "val": []},
            "best_score_dict": {'all': {}},
        }
        # store metrics per sample
        self.sample_score_dict = {}
        # store samples
        self.sample_dict =  {}


    def learning_rate_scheduler(self):
        """
        Learning Rate Schedular
        TODO: need to find all of the optimizers within the model classes
        """
        self.train_state = self.model.learning_rate_scheduler(self.train_state)
    def early_stopping_scheduler(self):
        """
        Early Stopping Method
        """
        self.train_state["es_step"] += 1
        if self.train_state["es_step"] >= self.cfg["es_criteria"]:
            print("stopping early...")
            self.train_state["stop_early"] = True
    def update_train_state(self):
        """
        Summary
        - updates the training state
        - checks if the current epoch is the best one
        - saves the checkpoint of the "best" epoch
        - reports best metric to ray-tune

        :return:
        """

        ep = self.train_state["epoch_index"]
        chosen_score = self.train_state["score_dict"]["val"][-1]['all']['all'][self.cfg["metric"]]

        """ Set the score to default (in cfg.trainer) or best chosen score"""
        if self.cfg["metric"] in self.train_state["best_score_dict"]:
            best_chosen_score = self.train_state["best_score_dict"][self.cfg["metric"]]
        else:
            best_chosen_score = self.cfg["default_best_score"]

        assert self.cfg['better_op'] in [">", "<", ">=", "<="]  # needs to be one of these operators

        # is the chosen score better than the best_chosen_score + threshold
        if eval(f"{chosen_score} "
                f"{self.cfg['better_op']}"
                f" ({best_chosen_score} + {self.cfg['improvement_threshold']})"):
            # save model only if loss is absolutely less than previous best
            # dont consider threshold
            self.train_state["best_score_dict"] = deepcopy(self.train_state["score_dict"]["val"][-1]['all']['all'])
            best_chosen_score = self.train_state["best_score_dict"][self.cfg["metric"]]

            # Reset early stopping step with threshold
            self.train_state["es_step"] = 0
            self.train_state["reduce_lr_step"] = 0

            # SAVE checkpoint first to provide in the experiment state
            self.save_model()

            logged_metrics = {}
            for split, split_metrics_dict in self.train_state["score_dict"].items():
                logged_metrics.update({f"{split}.{k}": v for k, v in split_metrics_dict[-1]['all']['all'].items()})
                mlflow.log_metrics(metrics=logged_metrics, step=ep)
            
            self.experiment_manager.log_metrics(run_id=self.run_id, metrics=logged_metrics)

            #self.tracker.set_tag(self.cfg['mlflow']['run_id'], 'metrics', self.train_state["best_score_dict"])

            # report to tune if hyperoptimization is being used
        else:

            # update early stopping and learniing rate scheduler
            self.early_stopping_scheduler()
            self.learning_rate_scheduler()

        if self.cfg['experiment']['num_samples'] > 1:
            tune.report(loss=best_chosen_score)

        if  (self.cfg['verbose'] and self.cfg['experiment']['num_samples'] == 1):
            for split in self.train_state["score_dict"].keys():
                print(f"""{split.upper()}\n
                        Epoch: {self.train_state['epoch_index']})--> {self.cfg['metric']}
                        Best : {round_sig(best_chosen_score,3)} ES: {self.train_state['es_step']}/{self.cfg['es_criteria']},
                        Seconds Per Sample : {int(self.train_state['samples_per_second'][split])} 
                
        
                      """)
    def final_eval(self,splits = ["train", "val"], verbose=True):
        # final evaluation
        self.load_model()
        for split in splits:
            self.eval(split=split, save_samples=True)
        self.save_state()


    def process(self, params={}, **kwargs):
        self.resolve_params(params=params)
        self.train()

    def resolve_params(self, params):
        for k,v in params.items():
            dict_put(self.cfg,keys=k, value=v)

    def save_state(self):
        """

        Save the Configuration,  State and Samples

        """
        if self.cfg['save_state']:
            # save the samples path and delete the samples (to sve space)
            self.cfg['train_state'] = self.train_state
            self.experiment_manager.log_artifact(run_id=self.run_id, obj= self.cfg, artifact_path="config.pkl")
    def save_model(self):
        self.model.save() 
    def load_model(self):
        self.model.load()

    # custom wrapper over dataloader for additional block
    def save_sample_scores(self,
                            timescale_token_batch_sample_score_dict,
                            split):

        for timescale, token_batch_sample_score_dict in timescale_token_batch_sample_score_dict.items():
            for token, batch_state_dict in token_batch_sample_score_dict.items():
                for obj_key,obj in batch_state_dict.items():
                    # get a list of tensors
                    if isinstance(obj, torch.Tensor):
                        obj = deepcopy(obj.detach().cpu().numpy())

                    if dict_has(input_dict=self.sample_score_dict, keys=[timescale,token,split, obj_key]):
                        self.sample_score_dict[timescale][token][split][obj_key] =\
                            np.concatenate([self.sample_score_dict[timescale][token][split][obj_key],obj])
                    else:
                        dict_put(input_dict=self.sample_score_dict, keys=[timescale,token,split,obj_key], value= obj)

    def save_samples(self, timescale_token_sample_dict,
                     split="val",
                     ):
        # we only want to save the input columns
        # we want to save the input  of what each model is trying ot predict
        # TODO more memory efficient solution is to save a reference to each row (post processed labels)
        for timescale, token_sample_dict in timescale_token_sample_dict.items():
            for token, sample_dict in token_sample_dict.items():            
                for obj_key,obj in sample_dict.items():
                                      # get a list of tensors
                    if isinstance(obj, torch.Tensor):
                        obj = deepcopy(obj.detach().cpu().numpy())

                    if dict_has(input_dict=self.sample_dict,
                                              keys= [timescale,token,split, obj_key]):
                        self.sample_dict[timescale][token][split][obj_key] =\
                            np.concatenate([self.sample_dict[timescale][token][split][obj_key],obj])
                    else:
                        dict_put(input_dict=self.sample_dict, keys=[timescale,token,split,obj_key], value= obj)
    def update_score_dict(self, timescale_token_score_dict,
                          timescale_token_sample_score_dict):
        """
        :param
            score_dict: dictionary of running mean scores
            batch_score_dict: dictionary of tensors
            batch_size: batch size o nvthe batch_score_dict (REDUNDENT?)
        """
        for timescale, token_sample_score_dict in timescale_token_sample_score_dict.items():
            for token, sample_score_dict in token_sample_score_dict.items():
                num_token_samples = len(next(iter(sample_score_dict.values())))
                for k, v in sample_score_dict.items():
                    if isinstance(v, torch.Tensor):
                        if v.requires_grad:
                            v = v.detach()
                        if len(v.shape) == 0:
                            v = v.item()
                        else:
                            v = v.mean().item()

                    if math.isnan(v):
                        print(f"WARNING: TS{timescale}-C{token}-{k} IS NULL")
                        continue

                    
                    if dict_has(input_dict=timescale_token_score_dict, keys=[timescale, token, k]):
                        # update the existing running mean counter
                        timescale_token_score_dict[timescale][token][k].update(value=v, count=num_token_samples)
                        timescale_token_score_dict["all"]["all"][k].update(value=v, count=num_token_samples)
                    else:
                        # create a new running mean counter
                        dict_put(input_dict=timescale_token_score_dict, keys=[timescale,token,k], value= RunningMean(value=v, count=num_token_samples))
                        dict_put(input_dict=timescale_token_score_dict, keys=["all","all",k], value=RunningMean(value=v, count=num_token_samples))

        return timescale_token_score_dict
    def train(self, split = 'train'):
        for epoch in range(self.cfg['num_epochs']):

            self.sample_dict = {}
            self.sample_score_dict = {}

            # stop early
            if self.train_state["stop_early"]:
                break
            # set each model into training mode
            self.model.train()

            # subdivide the batch size into seperate batches across different tokens
            
            t0 = time.time()
            sample_count = 0
            self.train_state["epoch_index"] = epoch
            timescale_token_score_dict = {}

            for idx in range(self.cfg['max_batches_per_epoch']):  # ALG STEP 2
                self.train_state["batch_index"] = idx  # increment the batch_idx state
                # choose a random token in the list (if there is more than 1)
                x_batch = ray.get(self.data.get_split_batch.remote(split=split))
                batch_size=list(x_batch.values())[0].shape[0]
                sample_count += batch_size
                # just in case the last batch is not equal to the batch size (drop==False)
                # a dicitonary to store tensors from anywhere
                pred_dict, sample_score_dict = self.model.learning_step(x_batch=x_batch) 

                #Update score dict

                timescale_token_score_dict = self.update_score_dict(timescale_token_score_dict=timescale_token_score_dict,
                                                    timescale_token_sample_score_dict=sample_score_dict)

   
            self.train_state['samples_per_second'][split] = sample_count/(time.time() - t0)



            self.train_state["score_dict"][split].append(dict_fn(input=timescale_token_score_dict, fn=self.get_mean))
            self.eval(split="val", save_samples=False)
            # update train state

            self.update_train_state()
        self.final_eval()

    @staticmethod
    def get_mean(v): 
        return v.value if isinstance(v,RunningMean) else v
    def eval(self, split="val", save_samples = False):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                split - the split for evaluation
                save_samples - whether to save samples (to save IO, only do it at the end)
            Return:
                None
        """
        # turn each model into eval mode
        # add dropout during evaluation

                
        self.model.eval()
        self.model.scale_dropout(scale_factor=self.cfg['eval_dropout_factor'])

        # calculate the total number of batches (round up just in case the last batch is partitioned)
        timescale_token_score_dict = {}
        t0 = time.time()
        sample_count = 0

        for idx in range(self.cfg['max_batches_per_epoch']):  # ALG STEP 2

            x_batch = ray.get(self.data.get_split_batch.remote(split=split, periods=self.cfg['data']['pipeline']['periods']))

            # adicitonary to store tensors from anywhere
            tensor_dict = {**x_batch} 
            batch_size = list(x_batch.values())[0].shape[0]

            pred_dict, sample_score_dict = self.model.evaluate(x_batch= x_batch, calculate_metrics=True)

            timescale_token_score_dict = self.update_score_dict(timescale_token_score_dict=timescale_token_score_dict,
                                                timescale_token_sample_score_dict=sample_score_dict)
            if save_samples:

                self.save_samples(timescale_token_sample_dict=pred_dict,
                                  split= split)
                self.save_sample_scores(timescale_token_batch_sample_score_dict=sample_score_dict,
                                        split=split)

            sample_count += batch_size


        self.train_state['samples_per_second'][split] =  sample_count/(time.time() - t0)
        self.train_state["score_dict"][split].append(dict_fn(input=timescale_token_score_dict, 
                                                                        fn=self.get_mean))
        self.model.train() 
        self.model.scale_dropout(scale_factor=self.cfg['eval_dropout_factor']) 
    def __del__(self):
        self.save_state()
    # custom wrapper over dataloader for additional block

