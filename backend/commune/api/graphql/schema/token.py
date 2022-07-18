
import graphene
from graphene import ObjectType, String, Date, List, Int, Float, ID,DateTime
from .experiment import Experiment



class CoinMeta(ObjectType):
    '''
    token meta info
    '''

    Name = String(required=True)

class CoinData(ObjectType):

    """
    model schema for token
    """
    # Direct Indicators (Add more if needed)
    Open = List(List(Float))
    Close = List(List(Float))
    High =  List(List(Float))
    Low =  List(List(Float))
    Volume = List(List(Float))
    # Time Features
    Timestamp = List(List(Int))
    Time = List(List(DateTime))

class SingleCoinSingleTimeScaleData(ObjectType):
    Name= String(required=True)
    TimeScale= String(required=True)
    Data= graphene.Field(CoinData)