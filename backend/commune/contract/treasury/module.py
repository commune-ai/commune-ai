
from copy import deepcopy
import sys
import brownie
import plotly.express as px

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

# st.set_page_config(layout="wide")

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
        self.contract.deposit(address, balance, name, mode, liquid ,{'from':self.account})
    
    def balanceOf(self, address:str, asset:str):
        return self.contract.balanceOf(address, asset)
    def myBalance(self, asset:str):
        address = self.account.address
        return self.balanceOf(address=address, asset=asset)

    def st_deposit(self):
        with st.sidebar.form('Deposit'):
            token = st.selectbox('Select a Token', list(token_address_map.keys()), 0)
            mode = st.selectbox('Select a Mode', ['ERC20', 'ERC721', 'TRSRY'], 0)
            balance = st.number_input('Select a Balance', 0)
            if st.form_submit_button("Submit"):
                self.deposit(token, mode, balance)


        with st.sidebar.form('WithDrawal'):
            token = st.selectbox('Select a Token', list(token_address_map.keys()), 0)
            mode = st.selectbox('Select a Mode', ['ERC20', 'ERC721', 'TRSRY'], 0)
            balance = st.number_input('Select a Balance', 0)
            if st.form_submit_button("Submit"):
                self.withdraw(token, mode, balance)
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


    def st_sidebar(self):

        self.st_deposit()


    def st_main(self):
        df = self.getAssets()
        
        with st.expander('Analytics', True):
            fig = px.pie(df, values='value', names='name', title='Asset Proportions')
            st.write(fig)
        with st.expander('Liquidity Proportion', True):
            fig = px.pie(df, values='value', names='liquid', title='Liquid Proportions')
            st.write(fig)

        with st.expander('Type Proportion', True):
            fig = px.pie(df, values='value', names='mode', title='Asset Mode Proportions')
            st.write(fig)


        address2index = {acc.address:i for i, acc in enumerate(self.accounts)}
        

        with st.sidebar.form('Network'):
            selected_address = st.selectbox('Select Account: ' , address2index, 0)
   
            selected_network = st.selectbox('Select Network: ' , ['local-fork', 'rinkeby', 'moonbeam'], 0)
            
            if st.form_submit_button("Submit"):
                self.setAccount(address2index[selected_address])

        
    def st_run(self):
        self.st_sidebar()
        self.st_main()
    
                
if __name__ == "__main__":

    tokens = token_address_map

    contract_module = ContractModule.deploy()
    contract_module.run()
    contract_module.simulation()


    contract_module.st_run()

    
    # print('FUCK')
    # print(portfolio.valueRatios)


    

    