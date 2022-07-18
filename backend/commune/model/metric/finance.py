import torch
import numpy as np
import streamlit as st
import pandas as pd
from copy import deepcopy
import datetime
import streamlit as st

from commune.transformation.block.categorical import NumericDiscretizer


def get_signal_df(pred_rel_diff,
                   gt_rel_diff,
                   timestamps,
                   random_baseline=False,
                   threshold=0.5,
                   timescale=1,
                   risk_scorer = None):
    """

    What is the roi across the time dimension, assuming

    Assumed Tensor Shape
        Batch Size x Time Dimension

    Note: These values are the relative change between the current and last time step

    """
    if isinstance(pred_rel_diff, np.ndarray):
        pred_rel_diff = torch.tensor(pred_rel_diff)

    if isinstance(gt_rel_diff, np.ndarray):
        gt_rel_diff = torch.tensor(gt_rel_diff)
    gt_rel_diff_start = gt_rel_diff[:, :1].repeat(1, gt_rel_diff.shape[1])

    if random_baseline:
        pred_rel_diff = pred_rel_diff[:,:1].repeat(1,pred_rel_diff.shape[1])

    pred_cummulative_difference = torch.cumprod(pred_rel_diff + 1, dim=1) - 1


    sample_id_tensor = torch.arange(pred_rel_diff.shape[0])[:,None].repeat(1,pred_rel_diff.shape[1])

    out_df_list = []

    for signal_type in ['sell','buy']:

        signal_factor = 1 if signal_type == 'buy' else -1

        pred_signal = torch.where(signal_factor*pred_cummulative_difference > 0, torch.ones_like(pred_cummulative_difference),
                                 torch.zeros_like(pred_cummulative_difference))

        gt_cummulative_difference = torch.cumprod(gt_rel_diff + 1, dim=1) - 1
        gt_signal = torch.where(signal_factor*gt_cummulative_difference > 0, torch.ones_like(gt_cummulative_difference),
                                 torch.zeros_like(gt_cummulative_difference))
        profit = (pred_signal * signal_factor*gt_cummulative_difference)

        # create a tensor with the time indexes per prediction
        temporal_positions = torch.arange(profit.shape[1])[None, :].repeat(profit.shape[0], 1).flatten() + 1

        profit = profit.flatten().squeeze(-1) # flatten the profit into
        gt_signal = gt_signal.flatten().squeeze(-1) # flatten the profit into
        pred_signal = pred_signal.flatten().squeeze(-1) # flatten the profit into
        gt = gt_cummulative_difference.flatten().squeeze(-1)
        gt_start = gt_rel_diff_start.flatten().squeeze(-1)

        pred = pred_cummulative_difference.flatten().squeeze(-1)
        sample_id_tensor = sample_id_tensor.flatten().squeeze(-1)
        correct_signal = (gt_signal == pred_signal).float()
        timestamps = timestamps.flatten().squeeze()
        signal_index = (profit != 0).flatten().nonzero().squeeze(-1)

        signal_timestamps = timestamps[signal_index]
        if isinstance(signal_timestamps, np.float32):
            signal_timestamps = [float(signal_timestamps)]
        elif isinstance(signal_timestamps, torch.Tensor):
            signal_timestamps = signal_timestamps.tolist()

        df = pd.DataFrame({'step': temporal_positions[signal_index].tolist(),
                             'profit': profit[signal_index].tolist(),
                             'pred_signal': pred_signal[signal_index].tolist(),
                             'pred': pred[signal_index].tolist(),
                             'gt_signal': gt_signal[signal_index].tolist(),
                             'gt': gt[signal_index].tolist(),
                             'gt_start': gt_start[signal_index].tolist(),
                             'correct_signal': correct_signal[signal_index].tolist(),
                             'signal_type': len(signal_index)*[signal_type],
                             'sample_id': sample_id_tensor[signal_index].tolist(),
                             'time': [datetime.datetime.fromtimestamp(int(ts)) for ts in signal_timestamps]})


        if len(df) > 0:
            df = df.sort_values(by=['time'])

            df['month'] = df['time'].dt.month
            df['year'] = df['time'].dt.year
            df['hour'] = df['time'].apply(lambda dt: dt.hour)
            df['weekday'] = df['time'].apply(lambda dt: dt.weekday())

            df['month_time'] = df['time'].apply(lambda dt: dt - datetime.timedelta(days=dt.day - 1,
                                            hours = dt.hour,
                                            minutes = dt.minute,
                                            seconds = dt.second))

            df['week_time'] = df['time'].apply(lambda dt: dt - datetime.timedelta(days=dt.isoweekday() % 7,
                                        hours = dt.hour,
                                        minutes = dt.minute,
                                        seconds = dt.second))

            df['step'] = df['step'].apply(lambda step: step * timescale)


            out_df_list.append(df)

    out_df = pd.concat(out_df_list)

    return out_df




def profit_over_time(df,
                   profit_period='month'):
    """

    What is the roi across the time dimension, assuming

    Assumed Tensor Shape
        Batch Size x Time Dimension

    Note: These values are the relative change between the current and last time step

    """
    period_df = deepcopy(df)

    period_df['time'] = period_df[f'{profit_period}_time']
    agg_function_dict = {
        'time':  lambda x: x.iloc[-1].date(),
        'profit_ratio': lambda x: x.iloc[-1],
        'step': lambda x: x.iloc[-1] + 1
    }

    final_df_list = []

    for period, period_df in period_df.groupby(['time', 'step']):
        period_df = period_df.sort_values(by=['time'])
        period_df['profit_ratio'] = (period_df['profit'] + 1).cumprod(skipna=True) - 1

        row_dict = {}
        for column, agg_fn in agg_function_dict.items():
            row_dict[column] = agg_fn(period_df[column])

        final_df_list.append(row_dict)


    final_df = pd.DataFrame(final_df_list).reset_index(drop=True)

    return final_df



class RiskStratifiedScore(object):
    def __init__(self,
                 numeric_columns=['gt_start'],
                 categorical_columns=['signal_type'],
                 score_column = 'profit'):

        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.group_columns = categorical_columns + numeric_columns
        self.score_column = score_column
        self.fit_bool = False

        self.numeric_descretizer_dict = {c: NumericDiscretizer(nbins=1000) for c in numeric_columns}

    def fit(self, input_df):

        df = deepcopy(input_df)

        for c in self.numeric_columns:

            df[c] = self.numeric_descretizer_dict[c].transform(df[c])

        df['score'] = df[self.score_column]

        self.score_group_df = df.groupby(self.group_columns).agg(np.mean).reset_index()[[*self.group_columns,'score']]
        self.fit_bool = True
        del df['score']


    def transform(self, df):

        df = deepcopy(df)


        if not self.fit_bool:
            self.fit(deepcopy(df))
        for c in self.numeric_columns:
            df[c] = self.numeric_descretizer_dict[c].transform(df[c])



        new_df = pd.merge(df, self.score_group_df, how='left', left_on=self.group_columns, right_on=self.group_columns)

        return new_df

