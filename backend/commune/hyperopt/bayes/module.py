import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from commune.process.base import BaseProcess
from commune.hyperopt.base import BaseHyperopt
from functools import partial
from copy import deepcopy


class BayesHyperopt(BaseHyperopt):
    default_cfg_path = f"{os.environ['PWD']}/commune/hyperopt/bayes/module.yaml"

    def __init__(self,cfg):
        super().__init__(cfg=cfg)

    def build_scheduler(self):
        default_cfg_scheduler=dict(
                        metric="loss",
                        time_attr='training_iteration',
                        mode="min",
                        max_t=5,
                        grace_period= 5,
                        reduction_factor=5)  
  
        cfg_scheduler = self.cfg.get('scheduler', default_cfg_scheduler)
        self.scheduler = ASHAScheduler(**cfg_scheduler)

    def build_reporter(self):
        default_cfg_reporter = dict(metric_columns = ["loss", "training_iteration"])
        cfg_reporter = self.cfg.get('reporter', default_cfg_reporter)
        self.reporter = CLIReporter(**cfg_reporter)

    def run(self, cfg, train_job, num_samples=1):
        self.build_scheduler()
        self.build_reporter()
        self.build_hyperparams(cfg=cfg)
        self.tune_config = {**self.hyperparams}
        return tune.run(partial(train_job, cfg=cfg),
                config=self.tune_config,
                progress_reporter=self.reporter,
                num_samples=num_samples,
                resources_per_trial=self.cfg['resources_per_trial'])





