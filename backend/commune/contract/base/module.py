import brownie
from brownie import network
from brownie import project
from commune.utils.misc import dict_put, get_object, dict_has
from commune.process import BaseProcess
from commune.contract.utils import setup_state
from copy import deepcopy
class ContractBaseModule(BaseProcess):
    abi = {}
    contract = None

    def __init__(self, cfg):
        BaseProcess.__init__(self, cfg=cfg)
        self.getEnvironment()
        
    @property
    def address(self):
        return self.contract.address
    
    def setAccount(self,key=0, mode="index", address=None,  *args, **kwargs):
        if mode == 'index': 
            assert isinstance(key, int), key

            print(self.accounts, key, 'DEBUG')
            self.account = self.accounts[key]
        else:
            raise NotImplementedError
        self.cfg['account']['address'] = self.account.address
        return self.account
    

    def getContract(self, address=None, name=None, args=[],  main_contract=True):
        contract = None
        if address == None:
            # deploy if address is None
            contract = self.deployContract(name=name, args=args)
        else: 
            try:
                contract = self.getDeployedContract(name=name, address=address)
            except brownie.exceptions.ContractNotFound:
                print(f"{name}:{address} Contract Not Found")
                contract = self.deployContract(name=name,args=args)
        # contract.set_alias(alias)

        if main_contract:
            self.abi = contract.abi  
            self.cfg['contract']['address'] = contract.address

        args = self.exclude_account_arg(args)

        return contract

    def ensure_acount_arg(self, args):
        '''
        ensure account is signing transaction
        '''
        assert isinstance(args, list)
        has_account = False
        for arg in args:
            if isinstance(arg, dict) and 'from' in arg:
                has_account = True

        if not has_account:
            args.append({'from': self.account})

        return args

    def change_state(self):
        for mode in ['write', 'read']:
            for obj_key in ['cfg', 'abi']:
                if dict_has(self.cfg, [mode, obj_key,'params','query']):
                    self.cfg[mode][obj_key]['params']['query'] = {"module": self.cfg['module'], "name": self.cfg.get('name')}
            
    def exclude_account_arg(self, args):
        '''
        ensure account is signing transaction
        '''
        has_account = False
        for i, arg in enumerate(args):
            if isinstance(arg, dict) and 'from' in arg:
                del args[i]

        return args

    def getDeployedContract(self,name, address ):
        if name in self.factory:
            return self.factory[name].at(address)    
        else:
            return self.network.contract.Contract(address)
        
    def deployContract(self, name, args=[]):
        assert name != None
        args = self.ensure_acount_arg(args=args )
        contract = self.factory[name].deploy(*args)
        return contract


    @staticmethod
    def getNetwork(network = "mainnet-fork", launch_rpc=True):

        """
        connects to neetwork, project/factory , and a account 
        """

        if not network.is_connected():
            network.connect(network=network, launch_rpc=launch_rpc)


        return network

        
    @staticmethod
    def getFactory():
        loaded_projects = project.get_loaded_projects()
        if len(loaded_projects)==0:
            factory = dict(project.load()) 
        else:
            factory = dict(loaded_projects[0])

        return factory

    @staticmethod
    def getAccounts():
        return list(network.accounts)


    def getEnvironment(self):

        self.cfg['network'] = self.cfg.get('network', dict(network = "mainnet-fork", launch_rpc=False))
        self.cfg['account'] = self.cfg.get('account', dict(key = 0, mode='index'))
        self.template_cfg = deepcopy(self.cfg)

        print(self.cfg['network'], 'BROOOO')

        self.network = self.getNetwork(**self.cfg['network'])
        self.factory = self.getFactory()
        self.accounts = self.getAccounts()
        self.setAccount(**self.cfg['account'])




    @property
    def other_accounts(self):
        return list(filter(lambda a: a.address != self.account.address, self.accounts))

    @property
    def function_abi_map(self):
        return {f_abi['name']:f_abi for f_abi in self.abi}
    @property
    def function_names(self):
        return list(self.function_abi_map.keys())


    def call(self, function, args=[]):
        if len(args) == 0:
            args.append({'from': self.account})
        output = getattr(self.contract, function)(*args)
        return self.parseOutput(function=function, outputs=output)


    def parseOutput(self, function, outputs):
        output_abi_list = self.function_abi_map[function]['outputs']
        
        parsedOutputs = {}
        for i,output_abi  in enumerate(output_abi_list) :
            output_key = i 
            if output_abi['name']:
                output_key = output_abi['name']
            
            parsedOutputs[output_key] = outputs[i]
            if 'components' in output_abi:
                component_names = [c['name'] for c in output_abi['components']]
                
                parseStruct = lambda o:  dict(zip(component_names, deepcopy(o)))
                if type(outputs[i]) in [list, tuple, set]:
                    parsedOutputs[output_key] = list(map(parseStruct, outputs[i]))
                else:
                    parsedOutputs[output_key] = parseStruct(outputs[i])
        
        return parsedOutputs


