
from app.backend.commune.hyperopt.base.module import HyperoptBase
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from commune.process.base import BaseProcess
import partial

class PBTHyperopt(HyperoptBase):

    def __init__(self,cfg):
        super().__init__(cfg=cfg)

    def run(self, cfg, train_job, num_samples=1):
        self.build_schedular() 
        self.build_reporter()
        self.tune_config = {**self.hyperparams}

        return tune.run(partial(train_job, cfg=cfg['trainer']),
                             config=self.tune_config,
                             scheduler=scheduler,
                             progress_reporter=reporter,
                             num_samples=cfg['num_samples'],
                             mode=self.cfg['mode'],
                            resources_per_trial= self.cfg['resources_per_trial'],
                             metric=self.cfg['metric'],
                             )



    def build_scheduler(self): 
        default_cfg_scheduler=dict(time_attr="training_iteration",
                                    perturbation_interval=5,
                                    hyperparam_mutations=self.tune_config)

        cfg_scheduler = self.cfg.get('scheduler', default_cfg_scheduler)
        self.scheduler = PopulationBasedTraining(**cfg_scheduler)


    def build_reporter(self):
        default_cfg_reporter = dict(
            metric_columns = ["loss", "training_iteration"]
        )

        cfg_reporter = self.cfg.get('reporter', default_cfg_reporter)
        self.reporter = CLIReporter(**cfg_reporter)


