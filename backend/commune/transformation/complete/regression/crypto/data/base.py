from commune.transformation.block.pandas import rolling_ma, temporal_difference
from commune.transformation.block.numpy import minmax_scalar
from commune.transformation.block.base import SequentialTransformPipeline
from commune.transformation.block.categorical import NaNLabelEncoder
from copy import deepcopy

import numpy as np

class DataTransformManager(object):
    """

    This Pipeline is responsible for creating additional columns and tranforming original
    columns on a split wise level

    """

    def __init__(self, config, process_pipeline_map=None):
        self.config = config
        self.process_pipeline_map = process_pipeline_map
        self.build()

    def transform(self, df):

        # we are changing the dictionary in the forloop
        transformed_df = deepcopy(df)

        for new_col, pipeline_args in self.process_pipeline_map.items():
            # Note the first time transform is called, it fits the functions
            # TODO: specifiy an argument to fit the functions?

            assert "input_column" in pipeline_args
            if pipeline_args["input_column"] in df:
                input_series = df[pipeline_args["input_column"]]

            # fit and transform onto training set data
            transformed_df[new_col] = pipeline_args["pipeline"].transform(input_series)

        return transformed_df

    def build(self):
        '''

        :param feature_group:
            map of group key to list of features
                - ie. technical: open, close, high, low
        :return: Builds Column2transform mapping keys to block pipeline
        '''

        if self.process_pipeline_map == None:

            process_pipeline_map = {}

            pipeline_template_dict = {
                # encode categorical features
                "cat": SequentialTransformPipeline(pipeline=[NaNLabelEncoder()]),

                # "con": SequentialTransformPipeline(pipeline=[
                #     # rolling_ma(window=self.config['pipeline']['ma_window'])
                # ]),
                # "con_diff": SequentialTransformPipeline(pipeline=[
                #     # rolling_ma(window=self.config['pipeline']['ma_window']),
                #     temporal_difference(lag=self.config['pipeline']['diff_step'])
                # ])
            }
            for group_name, feature_list in self.config['feature_group'].items():
                for col in feature_list:
                    if group_name == "cat":
                        process_pipeline_map[col] = {"pipeline": deepcopy(pipeline_template_dict[group_name]),
                                                      "input_column": col}
                    # elif group_name == 'cont':
                    #     process_pipeline_map[f"{col}_diff"] = {"pipeline": pipeline_template_dict[f'{group_name}_diff'],
                    #                                       "input_column": col}
                    #     self.config['feature_group'][group_name] += [f"{col}_diff"]
                    #
                    #     process_pipeline_map[col] = {"pipeline": pipeline_template_dict[group_name],
                    #                             "input_column": col}

            self.process_pipeline_map = process_pipeline_map