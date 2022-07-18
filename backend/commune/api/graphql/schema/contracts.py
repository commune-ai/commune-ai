

import graphene
from .ipfs import IPFSObject,IPFSObjectBundle
from .experiment import Experiment

# Batch of Predictions (Each prediciton would have a different start time)

class JSONState(graphene.ObjectType):
    state: graphene.JSONstring 
    timestamp: graphene.Time
    datetime: graphene.Datetime


class Contract(ObjectType):
    name= String(required=True)
    address=String(required=True)
    abi = graphene.JSONString
    abiPath=String
    state=JSONState

class ContractInput(graphene.InputObjectType):
    name = graphene.String(required=True)
    address = graphene.Int(required=True)
    abi = graphene.JSONString
    abiPath=graphene.String
    state=JSONState

class CreateContract(graphene.Mutation):
    class Arguments:
        contract_data = ContractInput(required=True)

    contract = graphene.Field(Contract)

    def mutate(root, info, contract_data=None):
        contract = Contract(
            name=contract_data.name,
            address=contract_data.address,
            abi=contract_data.address,
            abiPath=contract_data.address,
            state=contract_data.state
        )
        return CreateContract(contract=contract)