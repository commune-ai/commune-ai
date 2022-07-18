
from copy import deepcopy
import sys
import os
import datetime
sys.path.append(os.getenv('PWD'))
from commune.contract.model.portfolio import ContractModule as PortfolioModule
from commune.contract.market.portfolio import ContractModule as MarketModule

from .module import TasksModule





if __name__ == "__main__":

    accounts, network, factory = setup_state()

    portfolio = PortfolioModule(account=accounts[0], 
                                    network=network, 
                                    factory=factory)
                                    
    traderContract = portfolio.contract['Trader']

    tasksModule = TasksModule.setup()
    tasksModule.addMarketItem(traderContract.address)


    MarketModule()

    # print(portfolio.valueRatios)


    

    