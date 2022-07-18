

from torch import nn
from commune.model.metric import *
from commune.utils.misc import get_object
import torch
from commune.model.utils import scale_model_dropout
from commune.utils.misc import dict_put, dict_get
from commune.process.base import BaseProcess
from commune.experiment.manager.module import ExperimentManager
import mlflow
import os
import io


class GenericBaseModel(torch.nn.Module,BaseProcess):
    def __init__(self, cfg):
        BaseProcess.__init__(self, cfg=cfg)
        torch.nn.Module.__init__(self)
        self.scale_factor = 1
        self.client['experiment'] = ExperimentManager.deploy(actor=False, override={'client': self.client_manager})

    def forward(self, **kwargs):
        raise NotImplementedError("Implement the Forward Function Fam")


    def predict(self, **kwargs):
        return self(**kwargs)


    def learning_step(self, **kwargs):
        '''
        Calculate the learning step
        '''
        out_dict = self.predict(**kwargs)

        out_dict.update(kwargs)

        # calculate sample wise matrics
        sample_metrics = self.calculate_metrics(out_dict)
        self.optimizer.zero_grad()


        sample_metrics["total_loss"].mean().backward(retain_graph=True)
        self.optimizer.step()

        return out_dict,sample_metrics

    def learning_rate_scheduler(self, train_state):
        train_state["reduce_lr_step"] += 1
        if (train_state["reduce_lr_step"] >= train_state["reduce_lr_criteria"]):
            print("es hit, reducing LR...")
            for g in self.optimizer.param_groups:
                g["lr"] = g["lr"] * train_state["reduce_lr_factor"]
            train_state["reduce_lr_step"] = 0
        return train_state


    def calculate_metrics(self, x, optim_key="transformer"):
        """define you metrics that you will use"""

        score_dict = {"total_loss": 0}

        # lets start with the metrics and add them to the score dict
        for metric_key,metric_obj in self.metrics.items():
            # get the args from the x_args
            x_args = {k:x[v] for k,v in metric_obj["args"].items()}

            if "add_args" in metric_obj:
                x_args.update(metric_obj["add_args"])

            # get the loss from the function
            score_dict[metric_key] = metric_obj["fn"](**x_args)

            # only include the total loss if the 'w' (loss weight) is in the metric_obj and

            # the optim either equals optim_key or is not present in metric_obj (for only one optimizer)

            correct_optim_bool = ('optim' not in metric_obj) or \
                                  ('optim' in metric_obj and optim_key == metric_obj["optim"])

            if "w" in metric_obj and  correct_optim_bool:
                score_dict["total_loss"] += score_dict[metric_key] * metric_obj["w"]

        return score_dict


    def define_metrics(self):
        """calculate the metrics"""
        self.metrics = {}
        metric_bundle_fn = get_object(self.cfg['metric']['module'])
        metric_bundle_fn(metrics=self.metrics, cfg=self.cfg)

    def scale_dropout(self, scale_factor):
        self.scale_factor = scale_factor*self.scale_factor
        scale_model_dropout(model=self, scale_factor=scale_factor)

    @property
    def scaled(self):
        return bool(self.scale_factor != 1)

    def connect_data(self, cfg, data=None):

        '''
        Change the model configuration (cfg) based on any given state (data)

        returns  cls(cfg=cfg)
        '''

        raise NotImplementedError

    def get_outputs(self, input_dict, pipeline_map):
        '''
        get the useful outputs of the model from the raw predictions

        params:
            input_dict: dictionary of tesnors

            pipeline_map: map of pipeline transformations to tensors

        returns
            dictionary of useful tensors

        '''
        out_dict = {}

        NotImplementedError


        return out_dict



    def set(self, keys):
        self.setattr()
    def setattr(self, keys, value):
        dict_put(input_dict=self.__dict__, keys=keys, value=value)

    def get(self, keys):
        self.getattr()
    
    def getattr(self, keys):
        dict_get(input_dict=self.__dict__, keys=keys,)
    
    @staticmethod
    def resolve_path(path):
        path = path.split('//')[1]
        path_list = path.split('/')
        bucket_name = path_list[0]
        object_name = '/'.join(path_list[1:])
        return bucket_name, object_name

    def save(self):
        artifact_uri = self.cfg['experiment']['run']['artifact_uri']
        self.cfg['model_uri'] = os.path.join(artifact_uri, 'model.pth')
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        buffer.seek(0)
        self.client['minio'].put_object(path=self.cfg['model_uri'],
                                    data=buffer,
                                    length= len(buffer.getvalue()))
        
    
    def load(self):
        buffer = io.BytesIO(self.client['minio'].get_object(path=self.cfg['model_uri']).read())
        state_dict = torch.load(buffer)
        self.load_state_dict(state_dict)
        self.resolve_device()



    def resolve_device(self,device=None):
        # resolves the device
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == None:
            device = self.cfg.get('device', default_device)
        self.to(device) 


    