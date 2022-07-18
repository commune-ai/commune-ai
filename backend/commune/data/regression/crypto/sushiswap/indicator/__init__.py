from . import pine_indicators as pi
from . import update_pine_indicators as update_pi
from copy import deepcopy

class IndicatorProcessor(object):
    def __init__(self, r_offset=100):
        self.r_offset=100
    def process(self, df):

        return df

    def update(self,df, new_df):

        return new_df