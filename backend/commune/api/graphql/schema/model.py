
import graphene
from graphene import ObjectType, String, Date, List, Int, Float, ID,DateTime
from .experiment import Experiment

class Model(ObjectType):
    Name = String(required=True)
    Type = String(required=False)
    Experiments = List(Experiment)
    Coins = List(String)
