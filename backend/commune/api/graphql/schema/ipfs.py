

import graphene
from graphene import ObjectType, String, Date, List, Int, Float, ID,DateTime
from .experiment import Experiment

# Batch of Predictions (Each prediciton would have a different start time)


class IPFSObject(ObjectType):
    name=String()
    taxonomy=String()
    hash=String(required=True)
    object=String(required=True)


class IPFSObjectBundle(ObjectType):
    name=String(required=True)
    taxonomy=String()
    type=String(required=True)
    explainers= List(IPFSObject)
    hash= String(required=True)
