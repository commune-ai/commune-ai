

import graphene
from graphene import ObjectType, String, Date, List, Int, Float, ID,DateTime


class Run(ObjectType):
    id = ID(required=True)

class Experiment(ObjectType):
    id = ID(required=True)
    runs = List(Run)
