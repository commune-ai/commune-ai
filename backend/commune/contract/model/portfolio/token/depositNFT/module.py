
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
    default_cfg_path = f"{os.getenv('PWD')}/commune/contract/model/portfolio/token/depositNFT/module.yaml"


    def process(self, **kwargs):
        self.build()
        

    def build(self):

        # CREATE PORTFOLIO TRADER
        self.cfg['contract']['args'] = [self.cfg['name'], self.cfg['symbol']]
        self.contract = self.getContract(**self.cfg['contract'])
        
        print(self.cfg['contract'])
if __name__ == "__main__":

    contract_module = ContractModule.deploy()
    contract_module.run()
    print('FUCK')
    # print(portfolio.valueRatios)


    

    