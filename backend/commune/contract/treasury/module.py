
from copy import deepcopy
import sys
import brownie
import os
import pandas as pd
from typing import Type
sys.path.append(os.getenv('PWD'))
from commune.contract.address import dex_address_map, token_address_map
from commune.contract.base import ContractBaseModule
from commune.utils.misc import get_object
import random
from commune.contract.address import token_address_map
import datetime
import streamlit as st





class ContractModule(ContractBaseModule):
    default_cfg_path = f"contract.treasury.module"

    
    def process(self, **kwargs):
        self.build()
        self.token_address_map = token_address_map
        
    def build(self):

        # CREATE PORTFOLIO TRADER
        # self.cfg['contract']['args'] = [self.cfg['name']]
        self.contract = self.getContract(**self.cfg['contract'])



    def checkModeValidity(self, mode):
        assert mode in ['ERC721', 'ERC20', 'TRSRY']
    def isLiquidMode(self, mode):
        return bool(mode in ['ERC20', 'TRSRY'])


    def deposit(self, name:str, address:str,balance:int, mode:str):
        self.checkModeValidity(mode=mode)
        balance = brownie.Wei(balance)
        liquid = self.isLiquidMode(mode)
        st.write(address)
        self.contract.deposit(address, balance, name, mode, liquid ,{'from':self.account})
    
    def balanceOf(self, address:str, asset:str):
        return self.contract.balanceOf(address, asset)
    def myBalance(self, asset:str):
        address = self.account.address
        return self.balanceOf(address=address, asset=asset)


    def getAssets(self):

        asset_struct_fields = list(self.contract.assetStates(0).dict().keys())
        struct_outputs =  self.contract.getAssetStates()
        df = []
        for struct_output in struct_outputs:
            df.append({field: struct_output[i] for i,field in enumerate(asset_struct_fields)})

        return pd.DataFrame(df)



    def simulation(self):
        assets = [
            dict(address='0x55EE818aB469A814DaB0cfdC19B1717db00ACc3c',name='Shill Apes', mode='ERC721',  balance= random.randint(1,10)),
            dict(address='0x05F885aAE3fE4B92d58C6E89a5453d556B6080FD',name='Rent2Own Home Equity',mode='ERC721',  balance= random.randint(10,100)),
            dict(address='0xd0095DC7fDA212e5C049927BCD2950FC4B3e9F50',name='Zucks Toe in the Metaverse',mode='ERC721',  balance= random.randint(1,10)),
            dict(address=tokens['WETH'], name='WETH', mode='ERC20', balance=random.randint(1,10)),
            dict(address=tokens['DAI'], name='DAI', mode='ERC20', balance=random.randint(1,10)),
            dict(address='0x8cF09626dAA44c35712FE19E7eA1aEbfBf455bC9',name='Commune',mode='TRSRY',  balance= random.randint(1,10))
        ]

        list(map(lambda kwargs: self.deposit(**kwargs), assets))

        st.write(self.getAssets())

        
if __name__ == "__main__":

    tokens = token_address_map
    st.sidebar.write(tokens)

    contract_module = ContractModule.deploy()
    contract_module.run()
    contract_module.simulation()



    st.write(contract_module.getAssets())

    
    # print('FUCK')
    # print(portfolio.valueRatios)


    

    