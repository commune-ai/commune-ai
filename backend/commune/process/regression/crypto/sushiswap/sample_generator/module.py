from copy import deepcopy
import torch
import sys
import os
sys.path.append(os.environ['PWD'])

from commune.process import BaseProcess

import datetime
from commune.transformation.block.torch import step_indexer
from commune.utils.misc import get_object, timer, round_sig, dict_put, dict_get, dict_has, chunk
import random
import numpy as np
import gc
import pandas as pd
import itertools
from commune.transformation.block.hash import String2IntHash
import streamlit as st
def get_feature2group_map(group2features_map):
    """
    params:
        feature_groups: Dict[groip str, List[ feature str]]

    returns:
        Dict[feature str, group str]
    """
    feature2group_map = {}
    for group_key, feature_list in group2features_map.items():
        for feature_key in feature_list:
            feature2group_map[feature_key] = group_key
    return feature2group_map
def join_pipeline_maps(pipeline_maps):
    """

    Join Multiple Pipelines Together

    """
    pipeline_maps = list(map(deepcopy, pipeline_maps))
    left_pipeline_map = pipeline_maps[0]
    del pipeline_maps[0]

    for right_pipeline_map in pipeline_maps:
        for feature_key, right_pipeline in right_pipeline_map.items():
            if feature_key in left_pipeline_map:
                # since the data block occur prior, they need to preceed the sample block
                left_pipeline_map[feature_key].pipeline = [*left_pipeline_map[feature_key].pipeline,
                                                      *right_pipeline.pipeline]
            else:
                left_pipeline_map[feature_key] = right_pipeline
    return left_pipeline_map
class Process(BaseProcess):
    
    
    default_cfg_path = f"{os.environ['PWD']}/commune/config/process/regression/crypto/sushiswap/sample_generator/module.yaml"
    def change_state(self):
        self.cfg['write']['cfg']['params']['query']['token'] = self.cfg['token']

        self.table_name = {}
        self.table_name['processed_data'] = \
        self.cfg['read']["processed_data"]["params"]["table_name"].format(token=self.cfg["token"],
                                                                    base_ticker=self.cfg["base_ticker"],
                                                                    processed_data_tag= self.cfg['tag']["processed_data"]).lower()

    
        # get processed data objects
        self.cfg['read']["categorical_feature_info"]["params"]["object_name"] = \
            self.cfg['read']["categorical_feature_info"]["params"]["object_name"].format(
                token=self.cfg["token"],
                base_ticker=self.cfg['base_ticker'],
                processed_data_tag=self.cfg['tag']['processed_data'])

        self.cfg['read']["processed_data_pipeline_map"]["params"]["object_name"] = \
            self.cfg['read']["processed_data_pipeline_map"]["params"]["object_name"].format(
                token=self.cfg["token"],
                base_ticker=self.cfg['base_ticker'],
                processed_data_tag=self.cfg['tag']['processed_data']
                )

    def timestamp_bounds(self):
        timestamps = list(self.client['postgres'].query(
                f''' 
                    SELECT 
                        MAX(timestamp) as max_timestamp
                        MIN(timestamp) as min_timestamp
                    FROM {self.table_name["processed_data"]}
                ''',
                output_pandas=True)['timestamp'])
    def get_timestamps(self):
        timestamps = list(self.client['postgres'].query(
                f''' 
                    SELECT 
                        DISTINCT timestamp
                    FROM {self.table_name["processed_data"]}
                    WHERE expansion = 0
                ''',
                output_pandas=True)['timestamp'])

        return timestamps 
    @property
    def step_timestamp_size(self):
        ticker_amount = int(self.cfg["base_ticker"][:-1])
        ticker_abbr = str(self.cfg["base_ticker"][-1])
        
        if time_abbr == "m":
            step_size = 60*ticker_amount
        elif time_abbr == "h":
            step_sie = 3600*ticker_amount
        
        return step_size
    def split_timestamp_bounds(self, split):
        return list(map(lambda x: int(datetime.datetime.fromisoformat(x).timestamp()), self.cfg['splits'][split]))
    def split2timestamps(self, split):
        start_index = self.generated_periods['input']*self.max_step
        end_index = len(self.df_splits[split]) - self.generated_periods['output']*self.max_step



        
        return list(self.df_splits[split]['timestamp'][start_index:end_index])
    @property
    def split_keys(self):
        return list(self.cfg['splits'].keys())
    def generate_splits(self):
        df_splits = {}
        for split in self.split_keys:
            df =  self.get_raw_split( split=split)
            df['split'] = split
            df_splits[split] = df

        self.process_splits(deepcopy(df_splits))
        self.df_splits = df_splits
        self.df = pd.concat(list(self.df_splits.values()))

    def load_processors(self):
        self.processor = {}
        split_processor_class = get_object(f"commune.transformation.complete.regression.crypto.split.base.SampleTransformManager")
        self.processor['split'] = split_processor_class(feature_group=self.feature_group)

        sample_processor_class = get_object(f"commune.transformation.complete.regression.crypto.sample.diff.SampleTransformManager")
        self.processor['sample'] = sample_processor_class(feature_group=self.feature_group,
                                                periods=self.generated_periods,
                                                known_future_features=self.known_future_features)
    def process_splits(self,df_splits):
        # process splits
        for split, df_split in df_splits.items():
            df_splits[split] = self.processor['split'](df_split)

        return df_splits
    def get_raw_split(self, split="train"):
        df = self.generate_temporal_slice(self.split_timestamp_bounds(split=split))
        return df   

     
    def generate_temporal_slice(self,slice_timestamp_bounds=[]):    
        df = self.client['postgres'].query(
                f'''
                SELECT
                * 
                FROM {self.table_name["processed_data"]}
                WHERE timestamp >= {slice_timestamp_bounds[0]} AND 
                      timestamp <= {slice_timestamp_bounds[1]}
                ORDER BY "timestamp"''',
                output_pandas=True)

        return df

    def get_info(self):
        info_keys = [
            'input_columns',
            'input_dim',
            'periods',
            'generated_periods',
            'feature_group',
            'known_future_features',
            'categorical_feature_info'
        ]

        return {k:self.__dict__[k] for k in info_keys}



    def generate_meta(self):
        self.input_columns = list(itertools.chain.from_iterable(list(self.cfg['feature_group'].values())))
        self.input_dim = len(self.input_columns)
        self.feature_group = self.cfg['feature_group']
        self.known_future_features = self.cfg['known_future_features']
        self.token = self.cfg['token']
        self.generated_periods = self.cfg['generated_periods']
        self.periods = self.cfg['periods']
        self.meta_keys = self.cfg['meta_keys']
        self.base_ticker = self.cfg["base_ticker"]

        self.str2inthash = String2IntHash()
        self.max_step = int(self.cfg['max_ticker'][:-1]) // int(self.cfg['base_ticker'][:-1])
        # self.process_pipeline_map = self.cfg["process_pipeline_map"]


    def process(self,**kwargs):

        self.generate_meta()
        self.load_processors()
        self.generate_splits()
        # self.checks()
        # print(self.last_run_delay, "RUN_DELAY")

    def checks(self):
        print(f"DELAY IN MINUTES {self.token}", self.get_delay(),"FAGOGOOOOT")
    def get_delay(self, timescale = 'm'):
        most_recent_timestamp = self.df[self.df.extension == 0]['timestamp'].max()
        print("MAX DATETIME", datetime.datetime.fromtimestamp(most_recent_timestamp))
        most_recent_datetime = datetime.datetime.fromtimestamp(most_recent_timestamp).isoformat()
        
        if timescale == 'm':
            timescale_period = 60
        elif timescale == 's': 
            timescale_period = 1
        elif timescale == 'h': 
            timescale_period = 3600

        self.minutes_delayed =  (datetime.datetime.utcnow().timestamp() - most_recent_timestamp)//60
        return self.minutes_delayed

    @property
    def timestamps_vector(self):
        return torch.tensor(self.df[self.df['extension'] == 0]['timestamp'].to_numpy())

    def query_timestamp_index(self,query_timestamps):
        
        if isinstance(query_timestamps, list):
            query_timestamps = torch.tensor(query_timestamps)


        distance_matrix = torch.abs((self.timestamps_vector.unsqueeze(1) - query_timestamps.unsqueeze(0)))
        

        query_indices_tensor = torch.argmin(distance_matrix, dim=0)

        
        return query_indices_tensor


    def batch_generator(self,skip_step=10,timescales=['15m'], split="val", batch_size=32):
        get_batch_kwargs_list = []
        all_timestamps = self.split2timestamps(split)
        timestamp_samples = [v for i,v in enumerate(all_timestamps) if i % skip_step == 0 ]

        timestamp_batches = chunk(timestamp_samples,
                                chunk_size=batch_size,
                                append_remainder=False,
                                distribute_remainder=False,
                                num_chunks= None)
        for i,timestamp_batch in enumerate(timestamp_batches):

            for timescale in timescales:
               yield self.get_batch(timestamps=timestamp_batch, timescale=timescale)

        yield None

    def get_batch(self, timestamps=[], timescale=None, process_samples=True):

        periods = self.generated_periods
        if timescale == None: 
            timescale = self.base_ticker
        
        assert int(timescale[:-1]) % int(self.base_ticker[:-1]) == 0, f"{timescale} must be devisible by base ticker ({self.base_ticker})"
        timescale_factor = int(int(timescale[:-1])/int(self.base_ticker[:-1]))
        step = timescale_factor

        local_index_dict = {
            'input': torch.arange(-periods['input']*step ,0, step ),
            'output': torch.arange(0,periods['output']*step ,step)
        }
        sample_indices = self.query_timestamp_index(timestamps).unsqueeze(1)
        index_dict = {
            'input': sample_indices+local_index_dict['input'].unsqueeze(0),
            'output': sample_indices+local_index_dict['output'].unsqueeze(0),
        }


        sample_indices = torch.cat([index_dict['input'],index_dict['output']], dim=1)
        
        (batch_size, sequence_length) = sample_indices.shape

        raw_sample_dict= {}
        input_dict = {}
        meta_dict = {}

        for k in self.input_columns + self.meta_keys:
            raw_sample_dict[k] = torch.tensor(self.df[k].to_numpy()).unsqueeze(0).repeat(batch_size,1)
            if k in self.meta_keys:
                meta_dict[k] = torch.gather(raw_sample_dict[k],1, sample_indices) 
            else:
                input_dict[k] = torch.gather(raw_sample_dict[k],1, sample_indices) 


        raw_gt_dict = {}
        for gt_key in self.cfg['gt_keys']:
            raw_gt_dict[f"gt_past_{gt_key}-raw"] = input_dict[gt_key][:,:periods['input']]
            raw_gt_dict[f"gt_future_{gt_key}-raw"] = input_dict[gt_key][:,periods['input']:]

        if process_samples:
            input_dict = self.process_samples(input_dict)
        
        sample_dict = {**input_dict, **meta_dict, **raw_gt_dict}
        # get the ground truth keys
        for gt_key in self.cfg['gt_keys']:
            sample_dict[f"gt_past_{gt_key}"] = sample_dict[gt_key][:,:periods['input']]
            sample_dict[f"gt_future_{gt_key}"] = sample_dict[gt_key][:,periods['input']:]


        sample_dict['id'] = index_dict['input'][:,-1]
        sample_dict['token_hash'] = torch.tensor(batch_size*[self.str2inthash.transform(self.token)])
        sample_dict['timescale_hash'] = torch.tensor(batch_size*[self.str2inthash.transform(timescale)])
        return sample_dict

    def process_samples(self, input_dict):
        input_dict = self.processor['sample'](input_dict)
        input_dict = self.processor['split'](input_dict)
        return input_dict
    
    def get_pipeline_map(self):
        if not hasattr(self, "pipeline_map"):
            full_pipeline_map_list = [{k:v['pipeline'] for k,v in self.processor['split'].process_pipeline_map.items()},
                        {k:v['pipeline'] for k,v in self.processor['sample'].process_pipeline_map.items()}]
            self.pipeline_map = join_pipeline_maps(full_pipeline_map_list)
        
        return self.pipeline_map


if __name__ == '__main__':
    
    process = Process.deploy(actor=False)
    process.run()

    
    
    
        