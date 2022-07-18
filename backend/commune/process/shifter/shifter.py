
from copy import deepcopy
import sys
import datetime
import os
sys.path.append(os.environ['PWD'])
from commune.config import ConfigLoader
from commune.ray.utils import kill_actor, create_actor
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
                        nan_check
                        )

from commune.transformation.block.hash import String2IntHash
from commune.ray import ActorBase
from commune.process import BaseProcess

class ProcessShifter(BaseProcess):
    default_cfg_path="process.base.shifter"
    def __init__(self, cfg):
        super().__init__(cfg)

    def process(self, **kwargs):
        self.module = self.get_object(module).deploy(actor=False)
        return getattr(self.module, fn)(**kwargs)
