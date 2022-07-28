import requests
import json
import datetime
import pandas as pd
import os
import sys

sys.path[0] = os.environ['PWD']
from commune.config import ConfigLoader
from commune.utils.misc import  chunk
from commune.utils import format_token_symbol
import ray
import random
from copy import deepcopy
# function to use requests.post to make an API call to the subgraph url
from commune.process import BaseProcess
from tqdm import tqdm

from commune.ray.utils import kill_actor, create_actor
from ray.util.queue import Queue
import itertools
from commune.process.extract.crypto.utils import run_query



class Process(BaseProcess):

    """
    Manages loading and processing of data

    pairs: the pairs we want to consider
    base: the timeframe we consider buys on
    higher_tfs: all the higher timeframes we will use
    """
    default_cfg_path=f"{os.getenv('PWD')}/commune/extract/crypto/sushiswap/module.yaml"

    def change_state(self):
        self.cfg['write']['cfg']['params']['query']['token'] = self.cfg['token']

    @property
    def token(self):
        return self.cfg['token']


    @property
    def swap_token(self):
        return format_token_symbol(self.cfg['swap_token'],mode='sushiswap')

    @property
    def swap_tokens(self):
        swap_tokens = []
        for token in self.cfg['swap_tokens']:
            swap_tokens.append(format_token_symbol(token,mode='sushiswap'))
        return swap_tokens

    def table_name(self, table_name):
        table_name_dict = {}
        table_name_dict['sushiswap_data']= self.cfg['write']['sushiswap_data']['params']['table_name'].format(token=self.token, swap_token=self.swap_token).lower()
        table_name_dict['timestamp_block'] =  self.cfg['read']['timestamp_block']['params']['table_name']
        return table_name_dict[table_name]
    @property
    def timeBounds(self):
        return [
            datetime.datetime.fromisoformat(self.cfg['start_time']),
            datetime.datetime.fromisoformat(self.cfg['end_time']),
        ]

    @property
    def timestampBounds(self):
        return list(map(lambda x: x.timestamp(), self.timeBounds))


    def get_uncompleted_block_timestamp_map(self):


        completed_block_map = {}
        if self.client['postgres'].table_exists(self.table_name('sushiswap_data')) and \
                not self.cfg['refresh']:


            completed_block_list = self.client['postgres'].query(
            f'''SELECT DISTINCT(block) as completed_blocks FROM {self.table_name("sushiswap_data")}''', True)['completed_blocks']
            completed_block_map = {k:True for k in completed_block_list}

        min_block = self.query_min_block(tokens=[self.token, self.swap_token])
        
        timestamp_block_df = self.client['postgres'].query(
            f"""
            SELECT block, timestamp FROM 
            (SELECT 
                    block, 
                    timestamp,
                    row_number() OVER(ORDER BY block ASC) AS row
            FROM {self.table_name('timestamp_block')}
            WHERE timestamp >= {int(self.timestampBounds[0])} AND block >= {min_block}
            ) t
            WHERE t.row % {self.cfg['block_skip']}=0""",True)

        uncomplete_blocks = list(map(int,list(timestamp_block_df['block'])))
        uncomplete_timestamps = list(map(int,list(timestamp_block_df['timestamp'])))

        uncomplete_block_timestamp_map = {k:v for k,v in zip(uncomplete_blocks,uncomplete_timestamps )
                                            if (not completed_block_map.get(k, False)) or len(completed_block_map) == 0}

        return uncomplete_block_timestamp_map


    def process(self,  **kwargs):
        self.cfg.update(self.object_dict)
        
        if self.cfg['end_time'] == 'utcnow':
            self.cfg['end_time'] = (datetime.datetime.utcnow()).isoformat()

        if self.cfg['token'] == "WETH":
            self.cfg['swap_token'] = "DAI"
        else:
            self.cfg['swap_token'] = "WETH"


        if self.cfg['refresh']: 
            self.client['postgres'].delete_table(self.table_name("sushiswap_data"))

        # filter completed blocks
        self.block_timestamp_map =  self.get_uncompleted_block_timestamp_map()
        self.blocks = list(self.block_timestamp_map.keys())
        
        
        # self.blocks = self.blocks[:int(len(self.blocks) * self.cfg['block_sample_factor'])]
        
        self.get_data()

    def generate_unique_id(self, row):
        return row['id'][:5] + str(row['block'])
   


    def query_min_block(self, tokens):
        query = f"""
        {{
        s1: pairs(first: 1 orderBy: block, orderDirection: asc,  where: {{name_in: ['{tokens[0]}-{tokens[1]}']}}) 
            {{
            block
            }}
        s2: pairs(first: 1 orderBy: block, orderDirection: asc,  where: {{name_in: ['{tokens[1]}-{tokens[0]}']}}) 
            {{
            block
            }}
        }}
        """

        data = ray.get(run_query.remote(query,url=self.cfg['url']['sushiswap']))

        min_blocks = []
        for k,v in data.items():
            if v:
                min_blocks.append(v[0]['block'])
        
        return min(min_blocks)





    def query_template(self, block, tokens,swap_direction): 
        
        indicator_query_component = "\n".join(self.cfg['indicators'])
        query_template =  f"""t{block}_{swap_direction}: pairs( where: {{name_in:['{tokens[0]}-{tokens[1]}']}} ,
                            block: {{number:{block} }},
                                orderBy:reserveUSD,
                            orderDirection:desc) 
                    {{
                    id
                    token0 {{
                        symbol
                    }}
                    token1 {{
                        symbol
                    }}
                    {indicator_query_component}
                    
                }}
            """

        return query_template


        
    def generate_queries(self):

        query_cnt = 0
        query = "{"
        query_list = []
        
        assert self.swap_token != self.token, f"{self.swap_token}:{self.token}"
        for  block in self.blocks:
            query += self.query_template(block, [self.token, self.swap_token], swap_direction=0)

            query_cnt += 1
            if query_cnt % self.cfg['queries_per_worker']==0:
                query+= "}"
                query_list.append(query.replace("'", '"'))
                query= "{"
                
            query += self.query_template(block, [self.swap_token, self.token], swap_direction=1)
            query_cnt += 1

            if query_cnt % self.cfg['queries_per_worker']==0:
                query+= "}"
                query_list.append(query.replace("'", '"'))
                query= "{"


        if len(query) > 1:
            query+= "}"
            query_list.append(query.replace("'", '"'))


        return query_list
    def get_data(self):
        self.sushiswap_data = []
        running_query_list = []
        query_list = self.generate_queries()

        cnt = -1 
        for  query in query_list:
            cnt += 1
            running_query_list.append(run_query.remote(query,url=self.cfg['url']['sushiswap'])) 


        print(len(self.blocks), "BLOCKS LEFT")
        with tqdm(total=len(running_query_list)) as pbar:
            
            while running_query_list:
                finished_query_list, running_query_list = ray.wait(running_query_list)
                if len(finished_query_list) > 0:
                    sushiswap_data = []
                    query_data_list = ray.get(finished_query_list)
                    for query_data in query_data_list:
                        for raw_block_key, raw_block_pair_list in query_data.items():
                            for raw_block_pair in raw_block_pair_list:
                                pair_object = raw_block_pair
                                pair_block = int(raw_block_key[1:].split('_')[0])
                                swap_direction = int(raw_block_key[1:].split('_')[1])

                                pair_object['block'] = pair_block
                                pair_object['timestamp'] = self.block_timestamp_map[pair_block]
                                if swap_direction == 0:
                                    pair_object['token'] = pair_object['token1']['symbol']
                                    pair_object['direction'] = swap_direction
                                elif swap_direction == 1:
                                    pair_object['token'] = pair_object['token0']['symbol']
                                    pair_object['direction'] = swap_direction
                                else:
                                    print("FUCKED UP SWAP DATA")
                                    continue

                                del pair_object['token0'],pair_object['token1']
                                sushiswap_data.append(pair_object)
                    sushiswap_data = pd.DataFrame(sushiswap_data)
                    if len(sushiswap_data.columns) == 0:
                        continue



                    sushiswap_data['id'] = sushiswap_data.apply(lambda row: row['id'][:10]+str(row["block"]), axis=1)
                    for column in self.cfg['indicators']:
                        sushiswap_data[column] = sushiswap_data[column].astype(float)

                    self.client['postgres'].write_pandas_table(df=sushiswap_data,
                                                            table_name=self.table_name('sushiswap_data'),
                                                            refresh_table=self.cfg['refresh'],
                                                            primary_key='id')

                    if self.cfg['refresh']:
                        self.cfg['refresh'] = False
                    
                    pbar.update(len(finished_query_list))

            self.check()

    def check(self):
        count = self.client['postgres'].query(f"""SELECT count(*) from {self.table_name("sushiswap_data")}""")


        max_timestamp =  self.client['postgres'].query(f"""SELECT MAX(timestamp) as ts from {self.table_name("sushiswap_data")}""")[0][0]
        print(f"MAX DATETIME: {self.token} ", datetime.datetime.fromtimestamp(max_timestamp))
        print(f"{self.token}: Number of Rows {count}")
        distinct_pairs = self.client['postgres'].query(f"""
            SELECT DISTINCT token FROM {self.table_name("sushiswap_data")}
        """)
        print(f"DISTINCT PAIRS: ", distinct_pairs)
        # print(f"MISSED BLOCKS {len(blocks_left)}")



if __name__ == '__main__':

    sushiswap = Process.deploy()
    sushiswap.run()


