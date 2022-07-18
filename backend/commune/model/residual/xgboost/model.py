
import os
import mlflow
import streamlit as st
from commune.utils.misc import get_object
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from copy import deepcopy

class CompleteModel(object):
    cache={}
    def __init__(self,
                 cfg,
                 client):
        self.cfg = cfg
        self.client = client
        self.fit_bool = False


    def build_dag(self, data):

        self.target_feature = self.cfg['target']
        self.input_features = list(data.drop(columns=[self.cfg['target'], *self.cfg['ignore_features']]).columns)


        self.categorical_featues = self.cfg['categorical_features']
        self.numerical_features = [c for c in self.input_features  if c not in self.categorical_featues]




        numerical_transformer =Pipeline(steps=[('inputer',SimpleImputer(strategy='constant')),
                                                ('robust_scalar',RobustScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_featues)
            ])

        model = RandomForestRegressor(n_estimators=5, random_state=0)

        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', model)])


    def fit(self, data):
        self.build_dag(data)

        input = data[self.input_features]
        target = data[self.target_feature]


        self.model.fit(input, target)
        self.fit_bool = True

    def predict(self, data, pred_cache_key=None):
        input = data[self.input_features]
        pred = self.model.predict(input)
        if pred_cache_key is not None:
            self.cache[pred_cache_key] = pred

        data['score'] = pred
        return data

    def transform(self, data, threshold=None, predict=True, pred_cache_key=None):

        if predict:
            data = self.predict(data=data, pred_cache_key=pred_cache_key)

        if threshold:
            threshold_value =  pd.Series(self.cache[pred_cache_key]).quantile(threshold)
            data['include'] = data['score'] > threshold_value


        return data
    def load_model(self, run_id, refresh=False):
        run = mlflow.get_run(run_id)
        bucket_name = os.getenv('AWS_MLFLOW_BUCKET_NAME')
        run_artifact_dir_path = run.info.artifact_uri.replace(f"s3://{bucket_name}/", "")
        object_name = os.path.join(run_artifact_dir_path, 'explain', 'risk_model')

        model_state_dict= self.client['minio'].load(bucket_name=bucket_name,
                                           object_name=object_name,
                                            type='pickle')
        if model_state_dict is not None and not refresh:
            self.__dict__.update(model_state_dict)

    def save_model(self, run_id):

        run = mlflow.get_run(run_id)
        bucket_name = os.getenv('AWS_MLFLOW_BUCKET_NAME')
        run_artifact_dir_path = run.info.artifact_uri.replace(f"s3://{bucket_name}/", "")
        object_name = os.path.join(run_artifact_dir_path, 'explain', 'risk_model')
        from copy import deepcopy
        model_state_dict = {k:v for k,v in self.__dict__.items() if k not in ['client']}

        self.client['minio'].write(bucket_name=bucket_name,
                                  object_name=object_name,
                                  data=model_state_dict,
                                   type='pickle')