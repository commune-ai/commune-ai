
from copy import deepcopy
import sys
import os
from typing import Type
sys.path.append(os.getenv('PWD'))
from commune.contract.base import ContractBaseModule
import datetime

class ContractModule(ContractBaseModule):
    default_cfg_path = f"{os.getenv('PWD')}/commune/config/contract/market/base/module.yaml"

    def process(self):
        self.build()

    def build(self):
        self.cfg['contract']['args'] = [self.cfg['name'], {'from': self.account}]
        self.contract = self.getContract(**self.cfg['contract'])

    def addMarketItem(self, modelAddress):
        return self.contract.addMarketItem(modelAddress, {'from': self.account})

    def listMarketItems(self):
        return self.contract.listMarketItems()

    def removeMarketItem(self, itemId):
        return self.contract.removeMarketItem(itemId, {'from': self.account})




if __name__ == "__main__":

    ContractModule.deploy().run()

    # print(portfolio.valueRatios)


    

    