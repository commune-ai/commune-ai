

import graphene
from graphene import ObjectType, String, Date, List, Int, Float, ID,DateTime
from .experiment import Experiment

# Batch of Predictions (Each prediciton would have a different start time)
class PredictionData(ObjectType):
    token = String(required=True)
    mean = List(Float)
    upper = List(Float)
    lower = List(Float)
    gt = List(Float)
    gt_past = List(Float)
    timestamp_past = List(Float)
    timestamp = List(Int)
    datetime = List(DateTime)

class Swap(ObjectType):
    token0 = String(required=True)
    token1 = String(required=True)
    timescale = String(required=True)
    future_datetime = List(DateTime)
    current_datetime = List(DateTime)
    # period = Int(required=True)
    score = List(Float)
    signal = List(String)

class Ratio(ObjectType):
    tokens = List(String)
    ratios = List(Float)
    futureDatetime= DateTime(required=False)
    futureTimestamp = Int(required=True)
    currentDatetime= DateTime(required=False)
    currentTimestamp = Int(required=True)



