
from pycaret.regression import *
import os
import mlflow
from commune.utils.misc import get_object

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error




class CompleteModel(object):
    def __init__(self,
                 cfg,
                 client):
        self.cfg = cfg
        self.client = client
        self.fit_bool = False


    def fit(self, data):
        self.setup_env = setup(data=data, **self.cfg['setup'])
        self.model = blend_models(estimator_list=compare_models(**self.cfg['predict']['compare_models']))
        self.fit_bool = True

    def predict(self, data):
        if self.fit_bool == False:
            self.fit(data=data)
            out_data =  predict_model(self.model, data=data)
            self.train_data = out_data
        else:
            out_data = predict_model(self.model, data=data)
        out_data.rename(columns={'Label':'score'}, inplace=True)
        return out_data

    def transform(self, data, threshold=None, predict=True):

        if predict:
            data = self.predict(data=data)
        if threshold:
            threshold_value =  self.train_data['score'].quantile(threshold)
            data['include'] = data['score'] > threshold_value
        return data
    def load_model(self, run_id):
        run = mlflow.get_run(run_id)
        bucket_name = os.getenv('AWS_MLFLOW_BUCKET_NAME')
        run_artifact_dir_path = run.info.artifact_uri.replace(f"s3://{bucket_name}/", "")
        object_name = os.path.join(run_artifact_dir_path, 'explain', 'risk_model')

        model_state = self.client['minio'].load(bucket_name=bucket_name,
                                           object_name=object_name,
                                            type='pickle')

        self.model = model_state

    def save_model(self, run_id):

        run = mlflow.get_run(run_id)
        bucket_name = os.getenv('AWS_MLFLOW_BUCKET_NAME')
        run_artifact_dir_path = run.info.artifact_uri.replace(f"s3://{bucket_name}/", "")
        object_name = os.path.join(run_artifact_dir_path, 'explain', 'risk_model')


        self.client['minio'].write(bucket_name=bucket_name,
                                  object_name=object_name,
                                  data=self.model,
                                   type='pickle')