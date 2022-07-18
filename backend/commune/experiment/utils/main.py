import os
import shutil
import tempfile
import pickle
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import shutil
import torch
from .cleanup import (delete_experiment)
import streamlit as st
import itertools
import traceback
import numpy as np 


def get_experiment(experiment_name,
                   refresh_expeirment,
                   db_config,
                   store_config):
    """
    Creates and Experiment
    :param config:
    :return:
    """


    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment:
        if refresh_expeirment:
            delete_experiment(experiments=[experiment],
                              db_kwargs=db_config,
                              store_kwargs=store_config)
            mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(name=experiment_name)



    else:
        mlflow.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment_by_name(name=experiment_name)

    return experiment

def log_any_artifact(obj, artifact_path, filename=""):
    assert filename
    with tempfile.TemporaryDirectory() as dir:
        tempfilepath = os.path.join(dir,filename)
        with open(tempfilepath, "wb") as tmp_file:
            pickle.dump(obj, tmp_file)
            mlflow.log_artifact(local_path=tempfilepath, artifact_path=artifact_path)
        with open(tempfilepath, "rb") as tmp_file:
            mlflow.log_artifact(local_path=tempfilepath, artifact_path=artifact_path)

def get_best_run(Experiment,
                 MetricName,
                 Ascending,
                 ModelName,
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
    client = MlflowClient()

    """
    FILTER EXPERIMENTS 
    """

    EXPERIMENT_LIST = [e for e in client.list_experiments() if e.tags]


    exp_filter_fn_dict = {
        "Experiment": lambda e: e.name in Experiment}
        #                       or CoinName is None

    for filter_key, filter_fn in exp_filter_fn_dict.items():
        EXPERIMENT_LIST = list(filter(filter_fn,EXPERIMENT_LIST))

    """
    FILTER RUNS
    """

    run_list =client.search_runs([e.experiment_id for e in EXPERIMENT_LIST])

    # Filter Runs Base on Coin and ModelName

    run_filter_fn_dict = {
        "ModelName": lambda r: 'model_name' in r.data.tags and ModelName.lower() == r.data.tags["model_name"].lower()
    }

    for filter_key, filter_fn in run_filter_fn_dict.items():
        run_list = list(filter(filter_fn, run_list))
        print(filter_key, len(run_list))

    run_df = pd.DataFrame([{**run.data.params, **run.data.metrics} for run in run_list])

    best_run_index = run_df.sort_values(by=[MetricName],
                                        ascending=Ascending).index[index]
    best_run = run_list[best_run_index]
    if return_id:
        return best_run.info.run_id
    return best_run






def experiment_selection(default_experiment_name="DEMO_MULTI"):
    ################# SELECT EXPERIMENT #################################

    #### SIDEBAR
    # point to the mlflow tracking uri
    SAMPLES_DICT = None
    client = MlflowClient()
    EXPERIMENTS = [e for e in client.list_experiments() if e.tags]
    EXPERIMENT_NAME2ID = {e.name: e.experiment_id for e in EXPERIMENTS}
    # select a set of experiments to filter from
    ALL_EXPERIMENT_NAMES = list(set(EXPERIMENT_NAME2ID.keys()))




    # filter the experiments based on the experiment id

    st.sidebar.write(
    """
    ## SELECT A EXPERIMENT
    """
    )

    with st.sidebar.expander('Experiments'):
        EXPERIMENT_NAMES = st.multiselect("Filter By Experiments", ALL_EXPERIMENT_NAMES,[default_experiment_name])
        EXPERIMENT_IDS =  [EXPERIMENT_NAME2ID[name] for name in EXPERIMENT_NAMES]
        EXPERIMENTS = [client.get_experiment(e_id )for e_id in EXPERIMENT_IDS]
    # output mlflow runs as dict
    RUNS_DF = mlflow.search_runs(experiment_ids=EXPERIMENT_IDS,
                              filter_string="attribute.status != 'RUNNING'",
                              output_format="pandas")

    # Spefity the Model Type, the Model and the Model Prediction Type
    # Assumed Model Key Format : f'{Model_Type}.{Model_Name}_{Prediction_Type}'


    st.write(EXPERIMENT_IDS)


    with st.sidebar.expander("MODEL"):

        # Filter based on model type
        MODEL_TYPE_OPTIONS = list(set(RUNS_DF['tags.model_type'].tolist()))
        MODEL_TYPE = st.selectbox('Model Type', MODEL_TYPE_OPTIONS, 0)
        RUNS_DF = RUNS_DF[RUNS_DF['tags.model_type'].str.contains(MODEL_TYPE)]

        # filter based on model name
        MODEL_NAME_OPTIONS =  list(set(RUNS_DF['tags.model_name'].tolist()))
        MODEL_NAME = st.selectbox("Model Name", MODEL_NAME_OPTIONS, 0)
        RUNS_DF = RUNS_DF[RUNS_DF['tags.model_name'].str.contains(MODEL_NAME)]

        # filter based on prediction type
        MODEL_PREDICTION_TYPE_OPTIONS = list(set(RUNS_DF['tags.model_prediction_type'].tolist()))
        MODEL_PREDICTION_TYPE = st.selectbox("Model Prediction Type", MODEL_PREDICTION_TYPE_OPTIONS, 0)
        RUNS_DF = RUNS_DF[RUNS_DF['tags.model_prediction_type'].str.contains(MODEL_PREDICTION_TYPE)]


    ################# SELECT METRIC #################################


    # get the metric
    METRIC_PREFIX = "metrics."
    METRIC_OPTIONS = [r[len(METRIC_PREFIX):] for r in RUNS_DF.columns
                      if METRIC_PREFIX in r[:len(METRIC_PREFIX)]]
    # select a metric and the direction of ranking

    with st.sidebar.expander('Metric'):
        METRIC = st.selectbox("Metric", METRIC_OPTIONS, 0)

        METRIC = METRIC_PREFIX + METRIC

        RUNS_DF = RUNS_DF.dropna(subset=[METRIC])
        ASCENDING_METRIC_SORT = st.checkbox("Ascending Sort", False)

    BEST_RUN_ID = RUNS_DF.sort_values(by=[METRIC], ascending=ASCENDING_METRIC_SORT)['run_id'].iloc[0]


    return RUNS_DF, METRIC, ASCENDING_METRIC_SORT, BEST_RUN_ID







