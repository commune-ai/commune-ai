
from copy import deepcopy
import sys
import os
from typing import Type
sys.path.append(os.getenv('PWD'))
from commune.contract.address import dex_address_map, token_address_map
from commune.contract.base import ContractBaseModule
from commune.utils.misc import get_object
import datetime
import streamlit as st

class ContractModule(ContractBaseModule):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config/contract/model/portfolio/module.yaml"



    @property
    def valueRatios(self):
        token_value_ratio = {}
        for token_symbol in self.tokens.keys():
            token_value_ratio[token_symbol] = dict(self.contract.tokenStates(token_symbol))['ratio']
        return token_value_ratio

    @property
    def tokens(self):

        if self.cache.get('tokens'):
            return self.cache['tokens']
        
        tokens = {}
        for token_symbol in self.cfg['tokens']: 
            tokens[token_symbol] = self.getContract(name='ERC20',  
                                                    address = token_address_map[token_symbol])
        
        self.cache[cache_key] = tokens
        return self.cache[cache_key]

    def process(self, **kwargs):
        self.build()
        

    def build(self):

        # CREATE PORTFOLIO TRADER

        self.cfg['contract']['args'] = [self.cfg['name'], self.cfg['baseToken'], self.cfg['tokens'], self.get_token_address(self.cfg['tokens'])  ]

        self.contract = self.getContract(**self.cfg['contract'])

    
        self.module['contract']['NFT'].run(override={'cfg.name': self.cfg['name'], 
                                                  'cfg.symbol': self.cfg['symbol'], 
                                                    'cfg.account': self.cfg['account'],
                                                   'cfg.refresh': self.cfg['refresh']})
        # ADD NFT TO PORTFOLIO MANAGER
        if self.contract.depositNFT() != self.module['contract']['NFT'].address:
            self.contract.connectNFT(self.module['contract']['NFT'].address, {'from':self.account})

    def get_model_ratios(self):

        query = f'''
        {{ 
                            ratio(
                                        tokens: {list(self.tokens.keys())},
                                        timestamps: {[datetime.datetime.utcnow().timestamp()]},

                                        )
                                {{
                                    tokens
                                    ratios                                    
                                }}
        }}'''.replace("'", '"')
        response = self.client['graphql'].query(query=query)['ratio'][0]
        
        token_ratios = response['ratios']
        token_symbols = response['tokens']

        if not hasattr(self, "percentBase"):
            self.percentBase  = self.contract.percentBase()
        
        token_ratios = list(map(lambda x: int(x*self.percentBase), token_ratios))
        token_ratios_sum = sum(token_ratios)
        token_ratios = list(map(lambda x: ((x/token_ratios_sum)*self.percentBase), token_ratios))
        
        # assert token_ratios_sum ==  self.percentBase
        token_ratios_dict = dict(zip(token_symbols, token_ratios))

        self.predicted_token_ratios = token_ratios_dict
        return token_ratios_dict

    def rebalance(self, token_ratio_dict={},  callModel=True):
        
        if callModel:
            token_ratio_dict=self.get_model_ratios()
        token_symbols = list(token_ratio_dict.keys())
        token_ratios = list(token_ratio_dict.values())

        symbol_id_map = {symbol:i for i,symbol in enumerate(token_symbols)}
        # all the tokens need to be in the portfolio
        assert (len(token_symbols)>0)
        # sum of the portfolio needs to be 1

        if 'WETH' in symbol_id_map:
            del token_ratios[symbol_id_map['WETH']]
            del token_symbols[symbol_id_map['WETH']]

        tx = self.contract.rebalancePortfolio(token_symbols, token_ratios, True, {'from': self.account})
        
        return tx

    def get_token_address(self, tokens):
        return [token_address_map[token] for token in tokens]
    
    def addTokens(self, tokens):
        existing_tokens = list(self.contract.getTokens())
        tokens = list(filter(lambda t:t not in existing_tokens, tokens))
        token_address_list = [token_address_map[token] for token in tokens]
        tx = self.contract.addTokens(tokens, token_address_list, {'from': self.account})

    def removeTokens(self, tokens):
        existing_tokens = list(self.contract.getTokens())
        tokens = list(filter(lambda t:t not in existing_tokens, tokens))
        token_address_list = [token_address_map[token] for token in tokens]
        tx = self.contract.removeTokens(tokens, token_address_list, {'from': self.account})

    def setBaseToken(self,baseToken):
        # assert baseToken in self.contract.getTokens()
        self.contract.setBaseToken(baseToken, {'from': self.account})

    def addExplainer(self, module):
        """
        params:
            explainers=[{"name":"explainerName", "hash": "IPFShahs"}]
        """


        # for explain_key in explain_keys:


        #     query = f""" 
        #     {{
                
        #         explain(name:"{explain_key}", taxonomy:"{taxonomy}") {{
        #             name
        #             hash
        #         }}

        #     }}
            
        #     """

        #     explain_objects = self.client['graphql'].query(query)['explain']

        #     print(explain_key, explain_objects)

        #     for explain_object in explain_objects :
        #         self.contract['ExplainManager'].addExplain(explain_key, module , explain_object['hash'], {'from': self.account})

    @classmethod
    def getArgparse(cls, **kwargs):
        import argparse
        parser = argparse.ArgumentParser()
        # gets the
        parser.add_argument("--name", default=kwargs['name'], type=str, required=False, help="")
        parser.add_argument("--tag", default=kwargs['tag'], type=str, required=False, help="")
        parser.add_argument('--refresh', dest='refresh', action='store_true')

        input_args = parser.parse_args()

        return dict(input_args)


if __name__ == "__main__":

    contract_module_list = [ContractModule.deploy(actor=True, override={'name':f"Trader_{i}", 'actor.name': f"actor{i}" }) for i in range(1)]
    import ray
    ray.get([contract_module.run.remote() for contract_module in contract_module_list])

    # print(portfolio.valueRatios)


    

    