


import sys
import os
sys.path.append(os.getenv("PWD"))
import glob

import mlflow
from mlflow.tracking import MlflowClient
from commune.utils.misc import load_yaml
from commune.client.postgres.manager import PostgresManager
from commune.client.minio.manager import MinioManager
import psycopg2
import shutil
import torch


def remove_database_deleted_runs(db_kwargs):
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
    db = PostgresManager(db_kwargs)
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

    db.execute(query)
def remove_database_deleted_experiments(db_kwargs):
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
    db = PostgresManager(db_kwargs)
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
    db.execute(query, fetch=None)

def remove_deleted_experiment_artifacts(store_kwargs, experiment_id):
    '''
    Delete Expeirments in Postgres DB


    Args:
        client_kwargs:
              endpoint: 'localhost:9000'
              access_key: !ENV ${AWS_ACCESS_KEY_ID}
              secret_key: !ENV ${AWS_SECRET_ACCESS_KEY}
              secure: False

    '''

    store_client = MinioManager(store_kwargs)
    store_client.delete_folder('mlflow', experiment_id)

def remove_deleted_run_artifacts(store_kwargs, experiment_id, run_id):
    '''
    Delete Expeirments in Postgres DB


    Args:
        client_kwargs:
              endpoint: 'localhost:9000'
              access_key: !ENV ${AWS_ACCESS_KEY_ID}
              secret_key: !ENV ${AWS_SECRET_ACCESS_KEY}
              secure: False

    '''

    store_client = MinioManager(store_kwargs)
    store_client.delete_folder('mlflow', os.path.join(experiment_id, run_id))

def delete_experiment(experiments,
                      db_kwargs,
                      store_kwargs,
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
        db_kwargs['dbname'] = 'mlflow'
        remove_database_deleted_experiments(db_kwargs=db_kwargs)
        remove_deleted_experiment_artifacts(store_kwargs=store_kwargs,
                                            experiment_id=experiment.experiment_id)
        if verbose:
            print(f"{experiment.name} is deleted")


def delete_run(run,
               db_kwargs,
               store_kwargs):
    """

    :param run: run object
    """
    # delete run
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    mlflow.delete_run(run_id=run_id)

    # delete runs from postgres
    remove_database_deleted_runs(db_kwargs=db_kwargs)

    # delete artifacts from minio
    remove_deleted_run_artifacts(experiment_id=experiment_id,
                                 run_id=run_id,
                                 store_kwargs=store_kwargs
                                 )

### STILL IN PROGRESS ####

def remove_runs_by_tag(tag_key, tag_value):
    run_uuid_query = f"""
    SELECT run_uuid FROM tags WHERE key='{tag_key}' AND value='{tag_value}'
    """

    query = """
    DELETE FROM experiment_tags WHERE experiment_id=ANY(
         SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
    );
    DELETE FROM latest_metrics WHERE run_uuid=ANY(
        SELECT run_uuid FROM runs WHERE experiment_id=ANY(
            SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
        )
    );
    DELETE FROM metrics WHERE run_uuid=ANY();
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

    db.execute(query)

