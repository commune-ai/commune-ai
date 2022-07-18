from brownie import network
from brownie import project


import streamlit as st

def setup_state(network_name = "dev",
                launch_rpc=False 
                ):

    """
    connects to neetwork, project/factory , and a account 
    """

    if not network.is_connected():
        network.connect(network_name,launch_rpc)
    accounts = network.accounts
    
    loaded_projects = project.get_loaded_projects()
    if len(loaded_projects)==0:
        factory = dict(project.load()) 
    else:
        factory = dict(loaded_projects[0])



    return accounts, network, factory
