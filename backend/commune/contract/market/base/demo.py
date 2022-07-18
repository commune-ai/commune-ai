
from copy import deepcopy
import sys
import os
import datetime
sys.path.append(os.getenv('PWD'))

from .module import ContractModule


class DemoModule(ContractModule):

    def __init__(self, module):
        self.__dict__.update(module.__dict__)

    def run_tests(self):
        assert "test" in self.cfg['demo']
        task_cnt = len(self.cfg['demo']['test'])
        for i,test in enumerate(self.cfg['demo']['test']):
            test_function_key = f"test_{test}"
            assert hasattr(self, test_function_key), f"{test} not found"
            assert  callable(getattr(self, test_function_key)), f"{test} not a function"
            test_cfg = self.cfg['demo']['test'][test]
            
            
            if test_cfg['run'] == True:
                print(f"({i+1}/{task_cnt}) Running Test: {test.upper()}")
                getattr(self,test_function_key)(test_cfg=test_cfg)
            else:
                print(f"({i+1}/{task_cnt}) Skipping Test : {test.upper()}")
    
    def test_addModel(self, test_cfg):
        test_account = self.other_accounts[0]
        percentBase = self.contract['Trader'].percentBase()
        withdrawRatio = percentBase*test_cfg['withdrawRatio']
        
        print("Tester NFTS (before)", self.contract['NFT'].getAllOwnerTokenStates(test_account.address))
        print("Before Withdraw", self.tokens['WETH'].balanceOf(self.contract['Trader'].address))

        tx = self.contract['Trader'].withdrawNFT(withdrawRatio, {"from": test_account})
        # self.contract['Trader'].withdrawNFT({"value": deposit_value, "from": test_account})
        print("After Withdraw", self.tokens['WETH'].balanceOf(self.contract['Trader'].address))
        print("Tester NFTS (after)", self.contract['NFT'].getAllOwnerTokenStates(test_account.address))

        print("Owner State NFTS (after)", self.contract['NFT'].getOwnerState(test_account.address))

    def test_removeModel(self, test_cfg):


        for i in range(2):
            test_account = self.other_accounts[i]
            self.contract['Trader'].depositNFT({"value": test_cfg['deposit_value'], "from": test_account})
        print(self.contract['NFT'].getOwnerState(test_account.address, { "from": test_account}))

        # self.contract['Trader'].depositNFT({"value": deposit_value, "from": test_account})

    def test_swap(self, test_cfg):
        accounts = self.other_accounts
        transferAmount = accounts[1].balance() // 20

        for i in range(test_cfg['swap_rounds']):
            print(f"-- Round {i} --")
            tx = self.rebalance(callModel=True)
            for rebalance_event in tx.events['rebalanceEvent']:
                rebalance_event = dict(rebalance_event)
                token_decimals =  self.tokens[rebalance_event['symbol']].decimals()
                for k in ['initialValue', 'valueChange']:
                    rebalance_event[k] *= 10**-token_decimals  
            
            portfoio_ratios = deepcopy(self.valueRatios)
            for token, ratio in self.predicted_token_ratios.items():
                print(f'{token}: {int(ratio)} pred || {portfoio_ratios[token]} real')




if __name__ == "__main__":
    contract_module = ContractModule.setup()
    demo_module= DemoModule(contract_module)
    demo_module.run_tests()

       # print(portfolio.valueRatios)


    

    