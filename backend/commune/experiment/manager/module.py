import requests
import json
import datetime
import pandas as pd
import os
import sys
import torch
sys.path.append(os.environ['PWD'])
from commune.experiment.utils import get_best_run
from commune.process import BaseProcess
from commune.config import ConfigLoader
# function to use requests.post to make an API call to the subgraph url
from commune.ray.utils import kill_actor, create_actor
from commune.ray import ActorBase
import tempfile
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import ray

class ExperimentManager(BaseProcess, MlflowClient):
    """
    Manages loading and processing of data

    pairs: the pairs we want to consider
    base: the timeframe we consider buys on
    higher_tfs: all the higher timeframes we will use
    """
    default_cfg_path = f"{os.environ['PWD']}/commune/experiment/manager/module.yaml"
    cache = {'run': {}, 'config': {} }


    def __init__(self, cfg):
        BaseProcess.__init__(self, cfg=cfg)
        MlflowClient.__init__(self, **cfg['mlflow'])
        self.client['mlflow'] = MlflowClient(**cfg['mlflow'])

    def get_run_config(self, run_id=None):

        run = self.client['mlflow'].get_run(run_id)

        if dict_has(self.cache, keys=['config', run_id]):
            return self.cache['config'].get(run_id)

        minio_bucket = os.getenv('AWS_MLFLOW_BUCKET_NAME')
        run_artifact_dir_path = run.info.artifact_uri.replace(f's3://{minio_bucket}/', "")

        # get the artifact config to start up the client to connect with the artifact store
        # Load the Config Path

        config = self.client['minio'].load(bucket_name=minio_bucket, \
                                           object_name=os.path.join(run_artifact_dir_path, "config.pkl"))

        if config is not None:
            self.cache['config'][run_id] = config

        return config




    def get_best_run(self,
                 experiment,
                 metricName,
                 ascending,
                 modelName,
                 try_index_max=10,
                 return_config=True):



        get_run_kwargs = dict(
            experiment=experiment,
            metricName=metricName,
            ascending=ascending,
            modelName=modelName,

        )

        run_cache_key = '.'.join(list(map(str,list(get_run_kwargs.values()))))

        run_config = None

        if run_cache_key in self.cache['run'] :
            run =  self.cache['run'][run_cache_key]
            run_config = self.get_run_config(run=run)

        if run_config is None:
            get_run_kwargs['index'] = 0
            
            while get_run_kwargs['index'] < try_index_max:
                run = self._get_best_run(**get_run_kwargs)
                run_config = self.get_run_config(run=run)
                if run_config != None:
                    self.cache['run'][run_cache_key] = run
                    break
                get_run_kwargs['index'] += 1


        assert run_config is not None, 'run config is None'

        if return_config:
            return run ,run_config
        else:
            return run




    @property
    def experiments(self):
        return [e for e in self.client['mlflow'].list_experiments() if e.tags]

    @property
    def experiment_name2id(self):
        return {e.name: e.experiment_id for e in self.experiemnts}

    @property
    def experiment_names(self):
        return list(self.experiment_name2id.keys())

    @property
    def experiment_ids(self):
        return list(self.experiment_name2id.values())


    def remove_database_deleted_runs(self):
        query = """
        
        DELETE FROM latest_metrics WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs WHERE lifestage_cyle='deleted'
            );
        DELETE FROM metrics WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs WHERE lifestage_cyle='deleted'
            );
        DELETE FROM tags WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs WHERE lifestage_cyle='deleted'
            );
            
        DELETE FROM params WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs WHERE lifestage_cyle='deleted'
            );
        DELETE FROM runs WHERE  lifestage_cyle='deleted';
        """

        self.client['postgres'].execute(query)
    def remove_database_deleted_experiments(self):
        '''
        Delete Expeirments in Postgres DB


        Args:
            client_kwargs:
                host='localhost',
                port=5432,
                user='mlflow_user',
                password='mlflow_password',
                dbname='mlflow'

        R
        '''
        query = """
        DELETE FROM experiment_tags WHERE experiment_id=ANY(
            SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
        );
        DELETE FROM latest_metrics WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs WHERE experiment_id=ANY(
                SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
            )
        );
        DELETE FROM metrics WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs WHERE experiment_id=ANY(
                SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
            )
        );
        DELETE FROM tags WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs WHERE experiment_id=ANY(
                SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
            )
        );
        DELETE FROM params WHERE run_uuid=ANY(
            SELECT run_uuid FROM runs where experiment_id=ANY(
                SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
        ));
        DELETE FROM runs WHERE experiment_id=ANY(
            SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
        );
        
        DELETE FROM experiments where lifecycle_stage='deleted';
        """
        self.client['postgres'].execute(query, fetch=None)
    def remove_deleted_experiment_artifacts(self, experiment_id):
        '''
        Delete Expeirments in Postgres DB


        Args:
            client_kwargs:
                endpoint: 'localhost:9000'
                access_key: !ENV ${AWS_ACCESS_KEY_ID}
                secret_key: !ENV ${AWS_SECRET_ACCESS_KEY}
                secure: False

        '''

        self.client['minio'].delete_folder('mlflow', experiment_id)
    def remove_deleted_run_artifacts(self, experiment_id, run_id):
        '''
        Delete Expeirments in Postgres DB


        Args:
            client_kwargs:
                endpoint: 'localhost:9000'
                access_key: !ENV ${AWS_ACCESS_KEY_ID}
                secret_key: !ENV ${AWS_SECRET_ACCESS_KEY}
                secure: False

        '''

        self.client['minio'].delete_folder('mlflow', os.path.join(experiment_id, run_id))
    def delete_experiment(self, 
                        experiments,
                        verbose=False):
        '''
        Delete Experiment Store and DB Content

        params:
            experiment: experiment object or list of experiment objects
            db_kwargs: database client config
            store_kwargs: store client config

        TODO: Fix the meta data such that the index is reset to 0
        '''

        assert isinstance(experiments, list)

        for experiment in experiments:
            """DELETE EXPERIMENT if It is not Deleted"""
            if experiment.lifecycle_stage != "deleted":
                mlflow.delete_experiment(experiment.experiment_id)

            # remove experiment from client (Postgres/Mysql) and artifact store (Minio)
            self.remove_database_deleted_experiments()
            self.remove_deleted_experiment_artifacts(experiment_id=experiment.experiment_id)
            if verbose:
                print(f"{experiment.name} is deleted")
    def delete_run(self, run):
        """

        :param run: run object
        """
        # delete run
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        self.client['mlflow'].delete_run(run_id=run_id)

        # delete runs from postgres
        self.remove_database_deleted_runs()

        # delete artifacts from minio
        self.remove_deleted_run_artifacts(experiment_id=experiment_id,
                                    run_id=run_id)
    def get_experiment(self,name,refresh=False):

        experiment = self.client['mlflow'].get_experiment_by_name(name)

        if experiment:
            # experiment exists
            if refresh:
                self.delete_experiment(experiments=[experiment])
                self.client['mlflow'].create_experiment(name)
                experiment = self.client['mlflow'].get_experiment_by_name(name=name)



        else:
            self.client['mlflow'].create_experiment(name=name)
            experiment = self.client['mlflow'].get_experiment_by_name(name=name)

        return experiment
    def set_tags(self, run_id:str, tags:dict):
        for t_k, t_v in tags.items():
            self.client['mlflow'].set_tag(key=t_k, value=t_v, run_id=run_id)

    def log_experiment_tags(self, experiment_id:str, tags:dict):
        for t_k, t_v in tags.items():
            self.client['mlflow'].log_tag(key=t_k, value=t_v, experiment_id=experiment_id)

    def log_params(self,  run_id:str, params:dict):
        for p_k, p_v in params.items():
            self.client['mlflow'].log_param(key=p_k, value=p_v, run_id=run_id)
    def log_metrics(self, run_id, metrics:dict):
         for m_k, m_v in metrics.items():
            self.client['mlflow'].log_metric(key=m_k, value=m_v, run_id=run_id)       
    

    def log_artifact(self, run_id, obj, artifact_path='', filetype='pickle'):
        run =self.get_run(run_id=run_id)
        artifact_path = os.path.join(run.info.artifact_uri, artifact_path)
        self.client['minio'].write(path=artifact_path, data=obj,type=filetype)

    def get_run_info(self, run_id):
        return self.get_run(run_id).info
    def get_run_uri(self, run_id):
        return self.get_run(run_id).info.artifact_uri

    def save_state_dict(self, run_id, state_dict, artifact_path=None, **kwargs):
        with TempDir() as tmp:
            local_path = tmp.path()

            if not isinstance(state_dict, dict):
                raise TypeError(
                    "Invalid object type for `state_dict`: {}. Must be an instance of `dict`".format(
                        type(state_dict)
                    )
                )
            os.makedirs(local_path, exist_ok=True)
            state_dict_path = os.path.join(local_path, 'state_dict.pth')
            torch.save(state_dict, state_dict_path, **kwargs)
            self.log_artifact(run_id, local_path, artifact_path=artifact_path)

            if artifact_path:
                state_uri = os.path.join(self.get_run_uri(run_id), artifact_path, 'state_dict.pth')
            else:
                state_uri = os.path.join(self.get_run_uri(run_id), 'state_dict.pth')
            return state_uri

    def load_state_dict(self, artifact_path, **kwargs):
        """
        Load a state_dict from a local file or a run.

        :param state_dict_uri: The location, in URI format, of the state_dict, for example:

                        - ``/Users/me/path/to/local/state_dict``
                        - ``relative/path/to/local/state_dict``
                        - ``s3://my_bucket/path/to/state_dict``
                        - ``runs:/<mlflow_run_id>/run-relative/path/to/state_dict``

                        For more information about supported URI schemes, see
                        `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                        artifact-locations>`_.

        :param kwargs: kwargs to pass to ``torch.load``.
        :return: A state_dict

        .. code-block:: python
            :caption: Example

            with mlflow.start_run():
                artifact_path = "model"
                mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path)
                state_dict_uri = mlflow.get_artifact_uri(artifact_path)

            state_dict = mlflow.pytorch.load_state_dict(state_dict_uri)
        """

        with TempDir() as tmp:
            local_path = tmp.path()
            print(artifact_path, local_path, 'BROOO')
            local_path = _download_artifact_from_uri(artifact_uri=artifact_path, output_path=local_path)

            state_dict_path = os.path.join(local_path, 'state_dict.pth')
            return torch.load(state_dict_path, **kwargs)

    def _get_best_run(experiment,
                    metricName,
                    ascending,
                    modelName,
                    return_id =False,
                    index = 0):
        """
        params:
            "CoinName": token
            "ModelName": model key
            "MetricName": metric key
            "Ascending":  ascending sort of metric key for finding best run,
            "Experiment": list of experiments to include
        """

        """
        FILTER EXPERIMENTS 
        """
        experiment_list = [e for e in self.client['mlflow'].list_experiments() if e.tags]

        exp_filter_fn_dict = {
            "Experiment": lambda e: e.name in experiment}
            #                       or CoinName is None

        for filter_key, filter_fn in exp_filter_fn_dict.items():
            experiment_list = list(filter(filter_fn,experiment_list))

        """
        FILTER RUNS
        """

        run_list =self.client['mlflow'].search_runs([e.experiment_id for e in experiment_list])

        # Filter Runs Base on Coin and ModelName

        run_filter_fn_dict = {
            "modelName": lambda r: 'model_name' in r.data.tags and modelName.lower() == r.data.tags["model_name"].lower()
        }

        for filter_key, filter_fn in run_filter_fn_dict.items():
            run_list = list(filter(filter_fn, run_list))

        run_df = pd.DataFrame([{**run.data.params, **run.data.metrics} for run in run_list])

        best_run_index = run_df.sort_values(by=[metricName],
                                            ascending=ascending).index[index]
        best_run = run_list[best_run_index]
        if return_id:
            return best_run.info.run_id
        return best_run

