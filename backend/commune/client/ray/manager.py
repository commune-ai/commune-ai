from .object import RayObjectClient
from .queue import RayQueueClient
from commune.ray import ActorBase
import os
class RayManager(ActorBase):
    default_cfg_path=f"{os.environ['PWD']}/commune/config/client/block/ray.yaml"
    def __init__(self, cfg):
        self.queue = RayQueueClient(cfg=cfg['queue'])
        self.object = RayObjectClient(cfg=cfg['object'])

    
