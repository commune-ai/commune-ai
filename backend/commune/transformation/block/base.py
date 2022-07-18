
import numpy as np
import torch
import pandas as pd


class BaseTransform(object):
    '''
    fit: fit the transform
    forward/inverse: forward and inverse transform
    '''


    meta = {}

    def fit(self, data):
        pass

    def inverse(self, data):
        return data

    def transform(self, data):
        NotImplementedError("Implement Forward Function")



def input_schema_wrapper(process_type):

    """
    # Handles Torch Tensors, Pandas Series and Numpy Arrat

    params:
        process_type: The type that will be used to process the data


    """
    def decorator(func):
        def wrapper(self, x) :

            if isinstance(x, process_type):
                x = func(self,x)
            elif process_type == np.ndarray:
                if isinstance(x, torch.Tensor):
                    original_x_device = x.device
                    x = x.cpu().numpy()
                    x = func(self, x)
                    x = torch.tensor(x, device=original_x_device)

                elif isinstance(x, pd.Series):
                    x = x.to_numpy()
                    x = func(self, x)
                    x = pd.Series(x)

            elif process_type == torch.Tensor:
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x)
                    x = func(self, x)
                    x = x.cpu().numpy()

                if isinstance(x, pd.Series):
                    x = torch.tensor(x.to_numpy())
                    if len(x.shape) == 1:
                        x = x.unsqueeze(0)
                    x = func(self, x)
                    if len(x.shape) == 2:
                        x = x.squeeze(0)

                    x = pd.Series(x.cpu().numpy())
            return x

        return wrapper

    return decorator



class SequentialTransformPipeline(BaseTransform):
    def __init__(self, pipeline=[]):
        self.pipeline = pipeline
        assert pipeline, "The Pipeline is Empty"

    def transform(self, data):
        # fit then tranform
        for fn in self.pipeline:
            # Connect META dictionary with dictionary of pipeline
            # This is to transfer meta information between tranformations
            fn.meta = self.meta

            # transform data
            data = fn.transform(data)
        return data
    def inverse(self, data):
        for fn in reversed(self.pipeline):
            # if the function has an inverse transform then pass through it
            data = fn.inverse(data)
        return data


