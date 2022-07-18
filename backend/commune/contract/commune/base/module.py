
import sys
import os
from typing import Type
sys.path.append(os.getenv('PWD'))
from commune.contract.base import ContractBaseModule

class ContractModule(ContractBaseModule):
    default_cfg_path = f"{os.path.dirname(__file__)}/module.yaml"

    def process(self, **kwargs):
        self.build()

    def build(self):
        self.cfg['contract']['args'] = [self.cfg['name']]
        self.contract = self.getContract(**self.cfg['contract'])
        self.cfg['contract']['address'] = self.address
        print(self.address)

    def addItem(self, modelAddress):
        return self.contract.addItem(modelAddress, {'from': self.account})

    def listItems(self):
        return self.contract.listItems()

    def removeItem(self, itemId):
        return self.contract.removeItem(itemId, {'from': self.account})

if __name__ == "__main__":

    # ContractModule.deploy().run()
    print(os.path.dirname(__file__))


    

    