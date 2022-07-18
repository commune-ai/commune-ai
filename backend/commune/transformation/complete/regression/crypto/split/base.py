"""
Sample Wise Transformer
"""

import torch
from commune.transformation.block.base import SequentialTransformPipeline
from commune.transformation.block.torch import step_indexer, standard_variance, difference_transform

from copy import deepcopy

class SampleTransformManager(object):
    def __init__(self,
                 feature_group,
                 process_pipeline_map=None):


        self.__dict__.update(locals())

        if not self.process_pipeline_map:
            self.build()
    def build(self):
        """
        Build  mapping for tensor key to transform pipeline
        """
        process_pipeline_map = {}
        # price sample wise preprocessing for prices
        for key in self.feature_group['con']:
            process_pipeline_map[key] =  \
                {'pipeline':  SequentialTransformPipeline(pipeline=[
                     difference_transform(),
                    standard_variance()
                   
                    ]),
                'input_key': key}

        self.process_pipeline_map = process_pipeline_map

    def transform(self, x):

        """

        :param x: dictionary of tensors
        :return: dictionary of tensors post transformed
        """

        for key in x.keys():
            if key in self.process_pipeline_map:
                pipeline_state_dict = self.process_pipeline_map[key]
                input_key = pipeline_state_dict['input_key']
                pipeline_obj = pipeline_state_dict['pipeline']
                x[key] = pipeline_obj.transform(deepcopy(x[input_key]))
        return x

    def __call__(self, x):

        """
        :param kwargs: dicitonary of inputs
        :return: dictionary of outputs
        """

        with torch.no_grad():
            # prefix with gt_ if these keys are in predicted columns
            # we want to take the first initial input periods

            
            x = self.transform(x)

        return x


