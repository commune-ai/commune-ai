from copy import deepcopy
import os
import pandas as pd
import re
from commune.config.utils import resolve_devices, ConfigLoader
from commune.client import MinioManager
from commune.mlflow.tracking import MlflowClient



def get_best_runs(experiments=['TEST'],
                        metric='val.MSE_future_Close',
                        ascending=False,
                        num_best_runs=2):
    client = MlflowClient()

    # FILTER EXPERIMENT  #
    experiment_list = client.list_experiments()

    # filter by experiment name (list)
    experiment_list = list(filter(lambda e: e.name in experiments, experiment_list))

    # get a list of run objects
    run_list = client.search_runs([e.experiment_id for e in experiment_list])

    # Filter Runs Base on Coin and ModelName

    run_df = pd.DataFrame([{**run.data.params, **run.data.metrics} for run in run_list])

    best_run_index_list = list(run_df.sort_values(by=[metric], ascending=ascending).index)[:num_best_runs]

    best_run_ids = [run_list[i] for i in best_run_index_list]

    return best_run_ids


def get_ensemble_models(experiments=['TEST'],
                        model_patterns=['^GP'],
                        metric='val.MSE_future_Close',
                        ascending=False,
                        num_best_runs=2):
    client = MlflowClient()

    # FILTER EXPERIMENT  #
    experiment_list = client.list_experiments()

    # filter by experiment name (list)
    experiment_list = list(filter(lambda e: e.name in experiments, experiment_list))

    # get a list of run objects
    run_list = client.search_runs([e.experiment_id for e in experiment_list])

    run_list_model_options = list(
        set('.'.join([r.data.tags['model_name'], r.data.tags['model_prediction_type']]).lower() for r in run_list))

    # Filter Runs Base on Coin and ModelName

    best_run_per_model_dict = {}

    for model_pattern in model_patterns:

        # get model and prediction type

        matched_models = list(filter(lambda m: bool(re.match(model_pattern.lower(), m.lower())), run_list_model_options))

        print(matched_models)

        for model in matched_models:
            model_name, model_prediction_type = model.split('.')

            tmp_run_list = deepcopy(run_list)

            tmp_run_list = list(filter(lambda r: r.data.tags['model_name'].lower() == model_name.lower(), tmp_run_list))
            tmp_run_list = list(
                filter(lambda r: r.data.tags['model_prediction_type'].lower() == model_prediction_type.lower(),
                       tmp_run_list))

            assert len(tmp_run_list) > 0

            run_df = pd.DataFrame([{**run.data.params, **run.data.metrics} for run in tmp_run_list])

            best_run_index_list = list(run_df.sort_values(by=[metric], ascending=ascending).index)[:num_best_runs]

            best_run_per_model_dict[model] = [run_list[i] for i in best_run_index_list]

    return best_run_per_model_dict


