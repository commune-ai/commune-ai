from copy import deepcopy
import sys
import os
from typing import Type
sys.path.append(os.getenv('PWD'))
from commune.contract.model.portfolio import ContractModule as PortfolioModule
from commune.contract.market.portfolio import ContractModule as MarketModule
from commune.config import ConfigLoader
from commune.contract.utils import setup_state

if __name__ == "__main__":
     
    env_kwargs = MarketModule.getEnvironment()
    market = MarketModule.deploy(name="BORED APE SWAPERS", tag="v0", refresh=True, env_kwargs=env_kwargs)
    market.run()
    dummy_tags = list(map(lambda x: f"v{x}",list(range(8))))
    for i,tag in enumerate(dummy_tags):
        env_kwargs = MarketModule.getEnvironment(account_index=i)
        portfolio = PortfolioModule.deploy(name="ApeBot", tag=tag,env_kwargs=env_kwargs) 
        portfolio.run()
        portfolio.contract['Trader'].depositNFT({"value": "10 ether", "from": env_kwargs['account']})
        market.addMarketItem(portfolio.contract['Trader'].address)
    