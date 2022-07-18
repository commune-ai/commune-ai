import torch
import torch.nn as nn
import numpy as np
import re
from copy import deepcopy


class MonteCarloDropout(object):
    def __init__(self,
                 model = None,
                 num_samples=10,
                 output_mode = "bounds"):
        """
        wraps any model into a bayesian dropout model

        :param model: nn.Module
        :param num_samples: number of sampels
        :param pred_regex: regex expression
        :param output_mode:
            bounds: return mean, upper and lower bound
            std: retrun mean and standard deviation
        """
        self.model = model
        self.num_samples = num_samples
        self.output_mode = output_mode

    def __call__(self, **kwargs):
        out_dict = {}
        for i in range(self.num_samples):
            pred_dict = self.model(**kwargs)
            for sample_k, sample_v in pred_dict.items():
                if sample_k in out_dict:
                    out_dict[sample_k].append(sample_v)
                else:
                    out_dict[sample_k] = [sample_v]

        final_out_dict = {}
        for out_k, out_v in out_dict.items():
            out_v_stack = torch.stack(out_v, dim=0)
            out_v_mean = out_v_stack.mean(dim=0)
            out_v_std = out_v_stack.std(dim=0)
            
            if self.output_mode == "bounds":

                final_out_dict[f"{out_k}-mean"] = out_v_mean
                final_out_dict[f"{out_k}-upper"] = out_v_mean + out_v_std
                final_out_dict[f"{out_k}-lower"] = out_v_mean - out_v_std

            elif self.output_mode == "std":
                final_out_dict[f"{out_k}-mean"] = out_v_mean
                final_out_dict[f"{out_k}-std"] = out_v_std

            else:
                assert "your options of MC_outputs are [bounds(mean, upper, lower) and, std]"

        return final_out_dict




