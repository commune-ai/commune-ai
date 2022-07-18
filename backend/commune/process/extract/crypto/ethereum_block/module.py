import requests
import json
import datetime
import pandas as pd
import os
import sys
sys.path[0] = os.environ['PWD']
from commune.config import  ConfigLoader
from commune.utils.misc import dict_put, dict_has, chunk
import ray
import random
from copy import deepcopy
from commune.process import BaseProcess 
# function to use requests.post to make an API call to the subgraph url
from tqdm import tqdm
from commune.process.extract.crypto.utils import run_query

def get_ready_objects(running_jobs):
    ready_objects = []
    while running_jobs:
        ready_jobs, running_jobs = ray.wait(running_jobs)
        ready_objects.extend(ray.get(ready_jobs))

    return ready_objects

class Process(BaseProcess):
    """
    Manages loading and processing of data

    pairs: the pairs we want to consider
    base: the timeframe we consider buys on
    higher_tfs: all the higher timeframes we will use
    """
    default_cfg_path = f'{os.environ["PWD"]}/commune/process/extract/crypto/ethereum_block/module.yaml'

    def change_state(self):
        self.table_name = self.cfg['write']['timestamp_block']['params']['table_name']
        if self.cfg['end_time'] == 'utcnow':
            self.cfg['end_time'] = (datetime.datetime.utcnow()).isoformat()
    def process(self, **kwargs):
        self.get_block_times()

    def generate_query(self, timestamp_chunk):
        query = "{"
        for timestamp in timestamp_chunk:
            timestamp_bounds = [timestamp - self.cfg['timestamp_step']//2, 
                    timestamp +  self.cfg['timestamp_step']//2]
            query += f"""
              t{timestamp}:blocks(
                first: 1,
                orderBy: number,
                orderDirection: asc,
                where: 
                  {{
                    timestamp_gte: "{timestamp_bounds[0]}",
                    timestamp_lt: "{timestamp_bounds[1]}"
                  }}
              ) {{
                    number
              }}
            """

        query += "}"
        return query

    def roundTimestamp(self, timestamp):
        return int((timestamp//self.cfg['timestamp_step'])*self.cfg['timestamp_step'])
    


    @property
    def timeBounds(self):
        return [
            datetime.datetime.fromisoformat(self.cfg['start_time']),
            datetime.datetime.fromisoformat(self.cfg['end_time']),
        ]

    @property
    def timestampBounds(self):
        return list(map(lambda x: x.timestamp(), self.timeBounds))

    @classmethod
    def load_config(cls):
        config_path =f"{os.getenv('PWD')}/commune/config/extract/crypto/ethereum_block.yaml"
        return ConfigLoader(path=config_path, local_var_dict={}, load_config=True).cfg


    def get_last_completed_timestamp(self):

        last_completed_block = 0
        if self.client['postgres'].table_exists(self.table_name) and \
                not self.cfg['refresh']:

            last_completed_block = self.client['postgres'].query(
            f'''SELECT MAX("timestamp") FROM {self.table_name}''')[
            0][0]

        return int(last_completed_block)

    @classmethod
    def initialize(cls, cfg=None):
        if cfg is None:
            cfg = cls.load_config()

        return cls(cfg=cfg)


    def get_block_times(self):

        start_datetime = datetime.datetime.fromisoformat(self.cfg['start_time'])
        end_datetime = datetime.datetime.fromisoformat(self.cfg['end_time'])
        start_timestamp = self.roundTimestamp(start_datetime.timestamp())
        end_timestamp = self.roundTimestamp(end_datetime.timestamp())

        # filter already finished timestamps
        last_completed_timestamp = self.get_last_completed_timestamp()
        print("LAST COMPLETED ETH TIMESTAMP: ", datetime.datetime.fromtimestamp(last_completed_timestamp))
        start_timestamp = max(start_timestamp,last_completed_timestamp)
        end_timestamp = max(end_timestamp,last_completed_timestamp)
        assert end_timestamp >= start_timestamp, f"endtime: {end_timestamp} starttime: {start_timestamp}"
        timestamps = list(range(start_timestamp , end_timestamp, self.cfg['timestamp_step'])) 

        print("(ETH BLOCK MAP BUILDER )Timestamps to fetch: ",len(timestamps))

        
        if len(timestamps) == 0:
            return
        finished_query_list = []
        running_query_list = []

        timestamp_chunk_list = chunk(timestamps, chunk_size=self.cfg['queries_per_worker'])
        for timestamp_chunk in tqdm(timestamp_chunk_list):
            query = self.generate_query(timestamp_chunk=timestamp_chunk)
            running_query_list.append(run_query.remote(query, url=self.cfg['url']['ethereum_blocks']))

        with tqdm(total=len(running_query_list)) as pbar:
            while running_query_list:
                finished_query_list , running_query_list = ray.wait(running_query_list)
                if len(finished_query_list)> 0:

                    block_timestamp_df =  []
                    query_data_list = ray.get(finished_query_list)
                    for query_data in query_data_list:
                        for raw_timestamp_key, raw_block_object in query_data.items():

                            try:
                                block_object ={"timestamp": float(raw_timestamp_key[1:]),
                                            "block": float(raw_block_object[0]["number"])}
                                block_timestamp_df.append(block_object)
                            except IndexError:
                                print(raw_timestamp_key,': ', raw_block_object)
                    block_timestamp_df = pd.DataFrame(block_timestamp_df)

                    if len(block_timestamp_df)>0:
                        self.client['postgres'].write_pandas_table(df=block_timestamp_df,
                                                                table_name=self.table_name,
                                                                refresh_table=self.cfg['refresh'],
                                                                primary_key='block')
                        if self.cfg['refresh']:
                            self.cfg['refresh'] = False
                    pbar.update(len(finished_query_list))
if __name__ == '__main__':
    Process.deploy().run()
