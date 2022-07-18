
import os
from functools import partial
import numpy as np
import torch
from torch import nn

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList, MultitaskGaussianLikelihood
from .block import ExactGPModel
from commune.model.metric import *
import streamlit as st
EPS = 1E-10
class sequence_GP_Smoother(nn.Module):
    def __init__(self,
                 batch_size,
                 lr,
                 ):
        super().__init__()

        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_size]))
        self.gp = ExactGPModel(batch_size= batch_size,
                                     likelihood=self.likelihood )

        self.optimizer = torch.optim.Adam([
    {'params': self.parameters()}], lr=lr)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

    def fit(self, x, y):

        self.gp.train()

        with torch.autograd.set_detect_anomaly(True):
            for i in range(100):
                x = x.clone()
                self.optimizer.zero_grad()
                output = self.gp(x)


                loss = -self.mll(output, y).mean()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            print("\t", round(torch.abs(output.mean - y).mean().item(), 2))
    def inference(self, x):
        out_dict = {}
        self.gp.eval()
        with torch.no_grad():
            pred = self.likelihood(self.gp(x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict
    def transform(self, y, sample_freq=0.9):

        if isinstance(y, np.ndarray):
            y = torch.tensor(np.ndarray)

        self.gp.initialize()

        out_dict = {}

        batch_size = y.shape[0]
        full_period_steps = y.shape[1]
        input_period_steps = int(full_period_steps*sample_freq)

        x = torch.linspace(-1, 1, full_period_steps)\
                        .unsqueeze(0)\
                        .repeat(batch_size,1)\
                        .unsqueeze(2).to(self.args.trainer.device)

        x_rand_idx = torch.randperm(x.shape[1]).to(self.args.trainer.device)
        x = torch.index_select(x, 1, x_rand_idx )

        x_train = x[:, :input_period_steps]
        y_train = torch.index_select(y, 1, x_rand_idx )[:, :input_period_steps]


        self.gp.set_train_data(inputs=x_train, targets=y_train, strict=False)

        # fit
        self.fit(x=x_train, y=y_train)

        y_denoised = self.inference(x)["mean"]

        return y_denoised

class Sequence_GP_Extrapolator(nn.Module):
    def __init__(self,
                 batch_size,
                 lr,
                 steps,
                 verbose=False
                 ):
        super().__init__()
        self.steps = steps
        self.verbose = verbose

        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_size]))
        self.gp = ExactGPModel(batch_size= batch_size,
                                     likelihood=self.likelihood )

        self.optimizer = torch.optim.Adam([
    {'params': self.parameters()}], lr=lr)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

    def fit(self, x, y):

        self.gp.train()

        with torch.enable_grad():
            with torch.autograd.set_detect_anomaly(True):
                for i in range(self.steps):
                    x = x.clone()
                    self.optimizer.zero_grad()
                    output = self.gp(x)
                    loss = -self.mll(output, y).mean()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                if self.verbose:
                    print("\t", round(torch.abs(output.mean - y).mean().item(), 2))
    def inference(self, x):
        out_dict = {}
        self.gp.eval()
        with torch.no_grad():
            pred = self.likelihood(self.gp(x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict
    def forward(self, y, sample_freq=1.0, extension_periods=[0, 0]):
        """

        :param y: (b x N) time sequence of N elements

        :param sample_freq:
            - sampling frency of the line for fitting the gp
        :param extension_coeff:
            - ratio fo the sequence length fo extend
                ex: [0.2, 0.3] extends the left and right side by
                 20% and 30%

        :return:
            extended tensor of (b , N+ (sum(boundary_extension_factors)*N ))
        """

        y_mean, y_std = y.mean(), y.std()
        y = (y - y_mean) / (y_std + EPS)


        device = y.device


        self.gp.initialize()


        batch_size = y.shape[0]
        full_period_steps = y.shape[1]
        input_period_steps = int(full_period_steps*sample_freq)

        min_x = -1
        max_x = 1
        seq_len = max_x - min_x
        step_size = seq_len / full_period_steps

        x = torch.linspace(min_x, max_x, full_period_steps)\
                        .unsqueeze(0)\
                        .repeat(batch_size,1)\
                        .unsqueeze(2).to(device)

        x_rand_idx = torch.randperm(x.shape[1]).to(device)

        x_train = torch.index_select(x, 1, x_rand_idx )[:, :input_period_steps].to(device)
        y_train = torch.index_select(y, 1, x_rand_idx )[:, :input_period_steps].to(device)


        self.gp.set_train_data(inputs=x_train, targets=y_train, strict=False)
        # fit
        self.fit(x=x_train, y=y_train)

        x_extend = torch.arange(min_x - extension_periods[0]*step_size,
                                max_x +  extension_periods[1]*step_size,
                                step_size).to(device)

        y_extend= self.inference(x_extend)["mean"]

        y_extend  = y_extend*(y_std+EPS) + y_mean

        return y_extend
class oracle_GP(nn.Module):
    def __init__(self, args, model_args):
        super().__init__()

        self.model_args = model_args
        self.args = args

        batch_size = args.trainer.batch_size
        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_size]))
        self.gp = ExactGPModel(batch_size= batch_size,
                                     likelihood=self.likelihood )

        self.optimizer = torch.optim.Adam([
    {'params': self.parameters()}], lr=0.1)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

        self.define_metrics()

    def fit(self, x, y):

        self.gp.train()

        with torch.autograd.set_detect_anomaly(True):
            for i in range(50):
                x = x.clone()
                self.optimizer.zero_grad()
                output = self.gp(x)


                loss = -self.mll(output, y).mean()
                loss.backward(retain_graph=True)
                self.optimizer.step()

    def inference(self, x):
        out_dict = {}
        self.gp.eval()
        with torch.no_grad():
            pred = self.likelihood(self.gp(x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict


    def forward(self, **kwargs):

        out_dict = {}

        input_period_steps = self.args.data.periods.input
        output_period_steps = self.args.data.periods.output
        full_period_steps = input_period_steps +  output_period_steps

        batch_size = kwargs["index"].shape[0]
        x_full = torch.linspace(0, full_period_steps/input_period_steps, full_period_steps)\
                        .unsqueeze(0)\
                        .repeat(batch_size,1)\
                        .unsqueeze(2).to(self.args.trainer.device)


        x_s = x_full[:, -output_period_steps:]
        x_train = x_full[:, :input_period_steps]
        y_train = kwargs[self.model_args.predicted_columns[0]].squeeze(-1)

        self.gp.set_train_data(inputs=x_train.clone(), targets=y_train.clone(), strict=False)

        # fit
        self.fit(x=x_train, y=y_train)

        # infer
        gp_out_dict = self.inference(x=x_full)

        output_col = self.model_args.predicted_columns[0]
        out_dict[f"gp_pred_past_{output_col}-mean"] = gp_out_dict["mean"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-lower"] = gp_out_dict["lower"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-upper"] = gp_out_dict["upper"][:,:input_period_steps]

        out_dict[f"gp_pred_future_{output_col}-mean"] = gp_out_dict["mean"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-lower"] = gp_out_dict["lower"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-upper"] = gp_out_dict["upper"][:,-output_period_steps:]

        return out_dict

    def calculate_metrics(self, x):
        """define you metrics that you will use"""

        score_dict = {"total_loss": 0}

        # lets start with the metrics and add them to the score dict
        for metric_key,metric_obj in self.metrics.items():
            # get the args from the x_args
            x_args = {k:x[v] for k,v in metric_obj["args"].items()}

            if "add_args" in metric_obj:
                x_args.update(metric_obj["add_args"])

            # get the loss from the function
            score_dict[metric_key] = metric_obj["fn"](**x_args)
            if "w" in metric_obj:
                score_dict["total_loss"] += score_dict[metric_key] * metric_obj["w"]

        return score_dict


    def define_metrics(self):
        """calculate the metrics"""

        self.metrics = {}


        for output_name in self.model_args.predicted_columns:


            for mode in ["past", "future"]:
                self.metrics[f"MSE_gp_{mode}_{output_name}"]= \
                                                    dict(fn=partial(compute_mse, reduce_dims=[0,1]),
                                                        args=dict(y_pred=f"gp_pred_{mode}_{output_name}-mean",
                                                                  y_target=f"gt_{mode}_{output_name}"),
                                                        w= 1,
                                                        theChosenOne= False)

            """
            Map the get and pred to th 
            """
            for pair in self.args.data.transform.keys():
                output_transform_fn = self.args.data.transform[pair].column2transform[output_name]
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-lower"] = output_transform_fn

                self.args.data.transform[pair].column2transform[f"gt_future_{output_name}"] = output_transform_fn

                self.args.data.transform[pair].column2transform[f"gt_past_{output_name}"] = output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-lower"] =  output_transform_fn
class oracle_GRU(nn.Module):
    """
    LSTM model that takes in a sequence as input and predicts the number of times
    the patient will visit the hospital in the nect year. So output is a real-number (using ReLU)
    """

    def __init__(self, args, model_args):
        super().__init__()

        """RNN ENCODER"""
        self.rnn_enc = nn.GRU(
            input_size=args.data.input_dim,
            hidden_size=model_args.rnn_hidden_size,
            batch_first=True,
            dropout=model_args.dropout_rnn,
            bidirectional=False
        )

        """RNN DECODER"""
        self.rnn_dec = nn.GRU(
            input_size=model_args.output_dim,
            hidden_size=model_args.rnn_hidden_size,
            batch_first=True,
            dropout=model_args.dropout_rnn,
            bidirectional=False
        )

        """MULTIHEAD ATTENTION AFTER ENCODER WITH LAYER NORM"""

        if model_args.encoder_mh_attn_heads > 0:
            assert model_args.rnn_hidden_size % model_args.encoder_mh_attn_heads == 0, "MultiHeaded-Attention-Num Heads Aint Right"
            self.encoder_mh_attn = nn.MultiheadAttention(model_args.rnn_hidden_size, model_args.encoder_mh_attn_heads)

            self.encoder_mh_attn_alpha = nn.Parameter(torch.empty(1))
            self.encoder_mh_ln = nn.LayerNorm(model_args.rnn_hidden_size )


        """INFUSION DOT ATTENTION BETWEEN FUTURE ELEMENT AND EVERY PAST ENCODING"""
        if model_args.infuser_attn > 0:

            self.infuser_attn_ln = nn.LayerNorm(model_args.rnn_hidden_size)
            self.infuser_attn = nn.Linear(model_args.rnn_hidden_size + model_args.output_dim, args.data.periods.input)
            self.infuser_attn_combine = nn.Linear(model_args.rnn_hidden_size + ( model_args.rnn_hidden_size), model_args.rnn_hidden_size)



        self.act = nn.ELU()




        """FINAL SET OF LINEAR LAYERS"""
        self.h = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        for i in range(model_args.final_num_layers):
            if i == 0:
                in_dim_h = model_args.rnn_hidden_size
            elif i > 0:
                in_dim_h = out_dim_h
            out_dim_h = model_args.final_hidden_size // (2 ** (i+1))

            self.h.append(nn.Linear(in_dim_h, out_dim_h, bias=True))
            self.ln.append(nn.LayerNorm(out_dim_h))

        self.dropout_final = nn.Dropout(model_args.dropout_final)


        # OUTPUT LAYER
        self.out_c = nn.Linear(out_dim_h, model_args.output_dim, bias=True)




        self.args = args
        self.model_args = model_args

    def forward(self, **kwargs):

        out_dict = {}
        # we replicate the scalar features across every time-step

        x = torch.cat([kwargs[c].unsqueeze(-1).clone() for c in self.args.data.input_columns], dim=2)

        rnn_enc_out, h_enc = self.rnn_enc(x)

        if self.model_args.encoder_mh_attn_heads:
            rnn_enc_out_attn, _ = self.encoder_mh_attn(query=rnn_enc_out.permute(1, 0, 2),
                                           key=rnn_enc_out.permute(1, 0, 2),
                                           value=rnn_enc_out.permute(1, 0, 2))

            # LAYER NORMALIZATION
            rnn_enc_out = self.encoder_mh_ln(rnn_enc_out_attn.transpose(0,1) + rnn_enc_out)

        h_dec = rnn_enc_out.mean(1,keepdim=True).permute(1,0,2)
        future_steps=  self.args.data.periods.output



        h_dec_list = []
        z_dec_list = []

        out_future_list = []
        out_future_diff_list = []

        output_col = self.args.model.oracle.predicted_columns[0]

        for f in range(future_steps):

            pivot = kwargs[f"gp_pred_future_{output_col}-mean"][:,f].unsqueeze(1).unsqueeze(2).detach()

            if  self.training or f == 0:
                in_dec = pivot
            else:
                in_dec = out_future


            if self.model_args.infuser_attn:
                infuser_attn_weights = nn.Softmax(dim=-1)(self.infuser_attn(torch.cat([h_dec.squeeze(0), in_dec.squeeze(1)], -1)))

                """
                - attn_weights (BATCH_SIZE, MAX_SEQUENCE, MAX_SEQUENCE)
                - rnn_outs (B)

                """

                infuser_attn_applied = torch.bmm(infuser_attn_weights.unsqueeze(1), rnn_enc_out).permute(1, 0, 2)
                h_dec = self.infuser_attn_ln(self.infuser_attn_combine(torch.cat([infuser_attn_applied, h_dec], -1)) + h_dec)

            h_dec_list.append(h_dec.permute(1,0,2))
            rnn_dec_out, h_dec = self.rnn_dec(in_dec,h_dec)

            z = rnn_dec_out
            for i in range(len(self.h)):
                z = self.dropout_final(self.act(self.ln[i](self.h[i](z))))

            z_dec_list.append(z)

            out_future_diff = self.out_c(z)
            out_future = out_future_diff + pivot
            out_future_list.append(out_future)
            out_future_diff_list.append(out_future_diff)


        out_future = torch.cat(out_future_list, dim=1).squeeze(-1)
        out_future_diff = torch.cat(out_future_diff_list, dim=1).squeeze(-1)


        z_dec = torch.cat(z_dec_list, dim=1)
        z_enc  = rnn_enc_out

        for i in range(len(self.h)):
            z_enc = self.dropout_final(self.act(self.ln[i](self.h[i](z_enc))))
        out_past_diff = self.out_c(z_enc).squeeze(-1)


        out_dict[f"pred_past_{output_col}-mean"] = out_past_diff + kwargs[f"gp_pred_past_{output_col}-mean"].detach()
        out_dict[f"pred_past_{output_col}-lower"] = out_past_diff + kwargs[f"gp_pred_past_{output_col}-lower"].detach()
        out_dict[f"pred_past_{output_col}-upper"] = out_past_diff + kwargs[f"gp_pred_past_{output_col}-upper"].detach()

        out_dict[f"pred_future_{output_col}-mean"] = out_future_diff + kwargs[f"gp_pred_future_{output_col}-mean"].detach()
        out_dict[f"pred_future_{output_col}-lower"] = out_future_diff + kwargs[f"gp_pred_future_{output_col}-lower"].detach()
        out_dict[f"pred_future_{output_col}-upper"] = out_future_diff + kwargs[f"gp_pred_future_{output_col}-upper"].detach()
        return out_dict
class feature_extractor_GRU(nn.Module):
    """
    LSTM model that takes in a sequence as input and predicts the number of times
    the patient will visit the hospital in the nect year. So output is a real-number (using ReLU)
    """

    def __init__(self, args, model_args):
        super().__init__()

        """RNN ENCODER"""
        self.rnn_enc = nn.GRU(
            input_size=args.data.input_dim,
            hidden_size=model_args.rnn_hidden_size,
            batch_first=True,
            dropout=model_args.dropout_rnn,
            bidirectional=False
        )

        self.feature_dim = model_args.final_hidden_size // (2 ** (model_args.final_num_layers)),

        """RNN DECODER"""
        self.rnn_dec = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=model_args.rnn_hidden_size,
            batch_first=True,
            dropout=model_args.dropout_rnn,
            bidirectional=False
        )

        """MULTIHEAD ATTENTION AFTER ENCODER WITH LAYER NORM"""

        if model_args.encoder_mh_attn_heads > 0:
            assert model_args.rnn_hidden_size % model_args.encoder_mh_attn_heads == 0, "MultiHeaded-Attention-Num Heads Aint Right"
            self.encoder_mh_attn = nn.MultiheadAttention(model_args.rnn_hidden_size, model_args.encoder_mh_attn_heads)

            self.encoder_mh_attn_alpha = nn.Parameter(torch.empty(1))
            self.encoder_mh_ln = nn.LayerNorm(model_args.rnn_hidden_size )


        """INFUSION DOT ATTENTION BETWEEN FUTURE ELEMENT AND EVERY PAST ENCODING"""
        if model_args.infuser_attn > 0:

            self.infuser_attn_ln = nn.LayerNorm(model_args.rnn_hidden_size)
            self.infuser_attn = nn.Linear(model_args.rnn_hidden_size + self.feature_dim, args.data.periods.input)
            self.infuser_attn_combine = nn.Linear(model_args.rnn_hidden_size + ( model_args.rnn_hidden_size), model_args.rnn_hidden_size)



        self.act = nn.ELU()




        """FINAL SET OF LINEAR LAYERS"""
        self.h = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        for i in range(model_args.final_num_layers):
            if i == 0:
                in_dim_h = model_args.rnn_hidden_size
            elif i > 0:
                in_dim_h = out_dim_h
            out_dim_h = model_args.final_hidden_size // (2 ** (i+1))

            self.h.append(nn.Linear(in_dim_h, out_dim_h, bias=True))
            self.ln.append(nn.LayerNorm(out_dim_h))

        self.dropout_final = nn.Dropout(model_args.dropout_final)


        # OUTPUT LAYER

        self.args = args
        self.model_args = model_args

    def forward(self, x):

        rnn_enc_out, h_enc = self.rnn_enc(x)

        if self.model_args.encoder_mh_attn_heads:
            rnn_enc_out_attn, _ = self.encoder_mh_attn(query=rnn_enc_out.permute(1, 0, 2),
                                           key=rnn_enc_out.permute(1, 0, 2),
                                           value=rnn_enc_out.permute(1, 0, 2))

            # LAYER NORMALIZATION
            rnn_enc_out = self.encoder_mh_ln(rnn_enc_out_attn.transpose(0,1) + rnn_enc_out)
        z_enc = rnn_enc_out
        for i in range(len(self.h)):
            z_enc = self.dropout_final(self.act(self.ln[i](self.h[i](z_enc))))

        h_dec = rnn_enc_out.mean(1,keepdim=True).permute(1,0,2)
        z_dec_list = []

        for f in range(self.args.data.periods.output):

            if  self.training or f == 0:
                in_dec = z_enc[:,-1]
            else:
                in_dec = z


            if self.model_args.infuser_attn:
                infuser_attn_weights = nn.Softmax(dim=-1)(self.infuser_attn(torch.cat([h_dec.squeeze(0), in_dec.squeeze(1)], -1)))

                """
                - attn_weights (BATCH_SIZE, MAX_SEQUENCE, MAX_SEQUENCE)
                - rnn_outs (B)

                """

                infuser_attn_applied = torch.bmm(infuser_attn_weights.unsqueeze(1), rnn_enc_out).permute(1, 0, 2)
                h_dec = self.infuser_attn_ln(self.infuser_attn_combine(torch.cat([infuser_attn_applied, h_dec], -1)) + h_dec)

            rnn_dec_out, h_dec = self.rnn_dec(in_dec,h_dec)

            z = rnn_dec_out
            for i in range(len(self.h)):
                z = self.dropout_final(self.act(self.ln[i](self.h[i](z))))

            z_dec_list.append(z)

        z_dec = torch.cat(z_dec_list, dim=1)

        z = torch.cat([z_enc, z_dec], dim=1)

        return z
class oracle_GP_GRU(nn.Module):
    def __init__(self, args, model_args):
        super().__init__()

        self.model_args = model_args
        self.args = args

        self.model = {}
        self.optimizer = {}

        """Initialize the GP"""

        batch_size = args.trainer.batch_size
        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_size]))
        self.model["gp"] = ExactGPModel(batch_size= batch_size,
                                     likelihood=self.likelihood )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model["gp"])

        self.optimizer["gp"] = torch.optim.Adam([
    {'params': self.model["gp"].parameters()}], **model_args.gp.optimizer.__dict__)



        """Iniitalize the GRU"""

        self.model["gru"] = oracle_GRU(args=args, model_args=model_args.gru)
        self.optimizer["gru"] =  torch.optim.Adam([
    {'params': self.model["gru"].parameters()}], **model_args.gru.optimizer.__dict__)

        self.model = nn.ModuleDict(self.model)


        self.define_metrics()

    def gp_fit(self, x, y):

        self.model["gp"].train()
        with torch.autograd.set_detect_anomaly(True):
            self.optimizer["gp"]

            num_steps = 1 if self.training else 1
            for i in range(num_steps):
                x = x.clone()
                self.optimizer["gp"].zero_grad()
                output = self.model["gp"](x)

                loss = -self.mll(output, y).mean()
                loss.backward(retain_graph=True)
                self.optimizer["gp"].step()
    def gp_infer(self, x):
        out_dict = {}
        self.model["gp"].eval()
        with torch.no_grad():
            pred = self.likelihood(self.model["gp"](x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict


    def forward(self, **kwargs):

        out_dict = {}

        input_period_steps = self.args.data.periods.input
        output_period_steps = self.args.data.periods.output
        full_period_steps = input_period_steps +  output_period_steps

        batch_size = kwargs["index"].shape[0]
        x_full = torch.linspace(0, full_period_steps/input_period_steps, full_period_steps)\
                        .unsqueeze(0)\
                        .repeat(batch_size,1)\
                        .unsqueeze(2).to(self.args.trainer.device)


        x_train = x_full[:, :input_period_steps]
        y_train = kwargs[self.model_args.predicted_columns[0]].squeeze(-1)


        self.model["gp"].set_train_data(inputs=x_train.clone(), targets=y_train.clone(), strict=False)

        # fit gp

        self.gp_fit(x=x_train, y=y_train)

        # infer from gp
        gp_out_dict = self.gp_infer(x=x_full)

        output_col = self.model_args.predicted_columns[0]
        out_dict[f"gp_pred_past_{output_col}-mean"] = gp_out_dict["mean"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-lower"] = gp_out_dict["lower"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-upper"] = gp_out_dict["upper"][:,:input_period_steps]

        out_dict[f"gp_pred_future_{output_col}-mean"] = gp_out_dict["mean"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-lower"] = gp_out_dict["lower"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-upper"] = gp_out_dict["upper"][:,-output_period_steps:]

        """Pass the output of the gp into the gru"""
        gru_out_dict = self.model["gru"](**kwargs, **out_dict)

        out_dict.update(gru_out_dict)



        return out_dict

    def learning_step(self, **kwargs):
        out_dict = self(**kwargs)
        out_dict.update(kwargs)
        metrics = self.calculate_metrics(out_dict, optim_key="gru")

        self.optimizer["gru"].zero_grad()
        metrics["total_loss"].backward(retain_graph=True)
        self.optimizer["gru"].step()

        return out_dict,metrics


    def calculate_metrics(self, x, optim_key="gru"):
        """define you metrics that you will use"""

        score_dict = {"total_loss": 0}

        # lets start with the metrics and add them to the score dict
        for metric_key,metric_obj in self.metrics.items():
            # get the args from the x_args
            x_args = {k:x[v] for k,v in metric_obj["args"].items()}

            if "add_args" in metric_obj:
                x_args.update(metric_obj["add_args"])

            # get the loss from the function
            score_dict[metric_key] = metric_obj["fn"](**x_args)
            if "w" in metric_obj and (optim_key is not None and optim_key == metric_obj["optim"]):
                score_dict["total_loss"] += score_dict[metric_key] * metric_obj["w"]

        return score_dict


    def define_metrics(self):
        """calculate the metrics"""

        self.metrics = {}


        for output_name in self.model_args.predicted_columns:


            for mode in ["past", "future"]:
                self.metrics[f"MSE_gp_{mode}_{output_name}"]= \
                                                    dict(fn=partial(compute_mse, reduce_dims=[0,1]),
                                                        args=dict(y_pred=f"gp_pred_{mode}_{output_name}-mean",
                                                                  y_target=f"gt_{mode}_{output_name}"),
                                                        optim="gp")


                self.metrics[f"S1_{mode}_{output_name}"]= \
                                                    dict(fn=nn.SmoothL1Loss(),
                                                        args=dict(input=f"pred_{mode}_{output_name}-mean",
                                                                  target=f"gt_{mode}_{output_name}"),
                                                        w= self.model_args.loss_weight.mse,
                                                        optim="gru")

                self.metrics[f"MSE_{mode}_{output_name}"]= \
                                                    dict(fn=partial(compute_mse, reduce_dims=[0,1]),
                                                        args=dict(y_pred=f"pred_{mode}_{output_name}-mean",
                                                                  y_target=f"gt_{mode}_{output_name}"))

            """
            Map the get and pred to th 
            """
            for pair in self.args.data.transform.keys():
                output_transform_fn = self.args.data.transform[pair].column2transform[output_name]

                # pipeline for gt
                self.args.data.transform[pair].column2transform[f"gt_future_{output_name}"] = output_transform_fn
                self.args.data.transform[pair].column2transform[f"gt_past_{output_name}"] = output_transform_fn

                # pipeline for gp
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-lower"] = output_transform_fn

                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-lower"] =  output_transform_fn

                # pipeline for gp
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-lower"] = output_transform_fn


                self.args.data.transform[pair].column2transform[f"gt_past_{output_name}"] = output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-lower"] =  output_transform_f
class oracle_GP_GRU_V2(nn.Module):
    def __init__(self, args, model_args):
        super().__init__()

        self.model_args = model_args
        self.args = args

        self.model = {}
        self.optimizer = {}

        """Initialize the GP"""

        batch_size = args.trainer.batch_size
        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_size]))
        self.model["gp"] = ExactGPModel(batch_size= batch_size,
                                     likelihood=self.likelihood )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model["gp"])

        self.optimizer["gp"] = torch.optim.Adam([
    {'params': self.model["gp"].parameters()}], **model_args.gp.optimizer.__dict__)



        """Iniitalize the GRU"""

        self.model["gru"] = oracle_GRU(args=args, model_args=model_args.gru)
        self.optimizer["gru"] =  torch.optim.Adam([
    {'params': self.model["gru"].parameters()}], **model_args.gru.optimizer.__dict__)

        self.model = nn.ModuleDict(self.model)


        self.define_metrics()

    def gp_fit(self, x, y):

        self.model["gp"].train()
        with torch.autograd.set_detect_anomaly(True):
            self.optimizer["gp"]

            num_steps = 1 if self.training else 1
            for i in range(num_steps):
                x = x.clone()
                self.optimizer["gp"].zero_grad()
                output = self.model["gp"](x)

                loss = -self.mll(output, y).mean()
                loss.backward(retain_graph=True)
                self.optimizer["gp"].step()
    def gp_infer(self, x):
        out_dict = {}
        self.model["gp"].eval()
        with torch.no_grad():
            pred = self.likelihood(self.model["gp"](x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict


    def forward(self, **kwargs):

        out_dict = {}

        input_period_steps = self.args.data.periods.input
        output_period_steps = self.args.data.periods.output
        full_period_steps = input_period_steps +  output_period_steps

        batch_size = kwargs["index"].shape[0]
        x_full = torch.linspace(0, full_period_steps/input_period_steps, full_period_steps)\
                        .unsqueeze(0)\
                        .repeat(batch_size,1)\
                        .unsqueeze(2).to(self.args.trainer.device)


        x_train = x_full[:, :input_period_steps]
        y_train = kwargs[self.model_args.predicted_columns[0]].squeeze(-1)


        self.model["gp"].set_train_data(inputs=x_train.clone(), targets=y_train.clone(), strict=False)

        # fit gp

        self.gp_fit(x=x_train, y=y_train)

        # infer from gp
        gp_out_dict = self.gp_infer(x=x_full)

        output_col = self.model_args.predicted_columns[0]
        out_dict[f"gp_pred_past_{output_col}-mean"] = gp_out_dict["mean"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-lower"] = gp_out_dict["lower"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-upper"] = gp_out_dict["upper"][:,:input_period_steps]

        out_dict[f"gp_pred_future_{output_col}-mean"] = gp_out_dict["mean"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-lower"] = gp_out_dict["lower"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-upper"] = gp_out_dict["upper"][:,-output_period_steps:]

        """Pass the output of the gp into the gru"""
        gru_out_dict = self.model["gru"](**kwargs, **out_dict)

        out_dict.update(gru_out_dict)



        return out_dict

    def learning_step(self, **kwargs):
        out_dict = self(**kwargs)
        out_dict.update(kwargs)
        metrics = self.calculate_metrics(out_dict, optim_key="gru")

        self.optimizer["gru"].zero_grad()
        metrics["total_loss"].backward(retain_graph=True)
        self.optimizer["gru"].step()

        return out_dict,metrics


    def calculate_metrics(self, x, optim_key="gru"):
        """define you metrics that you will use"""

        score_dict = {"total_loss": 0}

        # lets start with the metrics and add them to the score dict
        for metric_key,metric_obj in self.metrics.items():
            # get the args from the x_args
            x_args = {k:x[v] for k,v in metric_obj["args"].items()}

            if "add_args" in metric_obj:
                x_args.update(metric_obj["add_args"])

            # get the loss from the function
            score_dict[metric_key] = metric_obj["fn"](**x_args)
            if "w" in metric_obj and (optim_key is not None and optim_key == metric_obj["optim"]):
                score_dict["total_loss"] += score_dict[metric_key] * metric_obj["w"]

        return score_dict


    def define_metrics(self):
        """calculate the metrics"""

        self.metrics = {}


        for output_name in self.model_args.predicted_columns:


            for mode in ["past", "future"]:
                self.metrics[f"MSE_gp_{mode}_{output_name}"]= \
                                                    dict(fn=partial(compute_mse, reduce_dims=[0,1]),
                                                        args=dict(y_pred=f"gp_pred_{mode}_{output_name}-mean",
                                                                  y_target=f"gt_{mode}_{output_name}"),
                                                        optim="gp")


                self.metrics[f"S1_{mode}_{output_name}"]= \
                                                    dict(fn=nn.SmoothL1Loss(),
                                                        args=dict(input=f"pred_{mode}_{output_name}-mean",
                                                                  target=f"gt_{mode}_{output_name}"),
                                                        w= self.model_args.loss_weight.mse,
                                                        optim="gru")

                self.metrics[f"MSE_{mode}_{output_name}"]= \
                                                    dict(fn=partial(compute_mse, reduce_dims=[0,1]),
                                                        args=dict(y_pred=f"pred_{mode}_{output_name}-mean",
                                                                  y_target=f"gt_{mode}_{output_name}"))

            """
            Map the get and pred to th 
            """
            for pair in self.args.data.transform.keys():
                output_transform_fn = self.args.data.transform[pair].column2transform[output_name]

                # pipeline for gt
                self.args.data.transform[pair].column2transform[f"gt_future_{output_name}"] = output_transform_fn
                self.args.data.transform[pair].column2transform[f"gt_past_{output_name}"] = output_transform_fn

                # pipeline for gp
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-lower"] = output_transform_fn

                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-lower"] =  output_transform_fn

                # pipeline for gp
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-lower"] = output_transform_fn


                self.args.data.transform[pair].column2transform[f"gt_past_{output_name}"] = output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-lower"] =  output_transform_fn
class oracle_GP_GRU_DKL(nn.Module):
    def __init__(self, args, model_args):
        super().__init__()

        self.model_args = model_args
        self.args = args

        self.model = {}
        self.optimizer = {}

        """Initialize the GP"""

        batch_size = args.trainer.batch_size
        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_size]))

        feature_extractor = feature_extractor_GRU(args=args, model_args=model_args.gru)
        self.model["gp"] = GP_DKL_Regressor(batch_size= batch_size,
                                     likelihood=self.likelihood,
                                     grid_size = self.args.data.periods.input // 2,
                                     feature_extractor= feature_extractor
                                     )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model["gp"])

        self.optimizer["gp"] = torch.optim.Adam([
    {'params': self.model["gp"].parameters()}], **model_args.gp.optimizer.__dict__)



        """Iniitalize the GRU"""

        self.model = nn.ModuleDict(self.model)


        self.define_metrics()

    def gp_fit(self, x, y):

        self.model["gp"].train()
        with torch.autograd.set_detect_anomaly(True):
            self.optimizer["gp"]

            num_steps = 1 if self.training else 1
            for i in range(num_steps):
                x = x.clone()
                self.optimizer["gp"].zero_grad()
                output = self.model["gp"](x)

                loss = -self.mll(output, y).mean()
                loss.backward(retain_graph=True)
                self.optimizer["gp"].step()
    def gp_infer(self, x):
        out_dict = {}
        self.model["gp"].eval()
        with torch.no_grad():
            pred = self.likelihood(self.model["gp"](x))
            out_dict["mean"] = pred.mean
            out_dict["lower"], out_dict["upper"] = pred.confidence_region()

        return out_dict


    def forward(self, **kwargs):

        out_dict = {}

        x = torch.cat([kwargs[c].unsqueeze(-1).clone() for c in self.args.data.input_columns], dim=2)

        input_period_steps = self.args.data.periods.input
        output_period_steps = self.args.data.periods.output
        full_period_steps = input_period_steps +  output_period_steps

        batch_size = kwargs["index"].shape[0]

        x_train = x_full[:, :input_period_steps]
        y_train = kwargs[self.model_args.predicted_columns[0]].squeeze(-1)


        self.model["gp"].set_train_data(inputs=x_train.clone(), targets=y_train.clone(), strict=False)

        # fit gp

        self.gp_fit(x=x_train, y=y_train)

        # infer from gp
        gp_out_dict = self.gp_infer(x=x_full)

        output_col = self.model_args.predicted_columns[0]
        out_dict[f"gp_pred_past_{output_col}-mean"] = gp_out_dict["mean"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-lower"] = gp_out_dict["lower"][:,:input_period_steps]
        out_dict[f"gp_pred_past_{output_col}-upper"] = gp_out_dict["upper"][:,:input_period_steps]

        out_dict[f"gp_pred_future_{output_col}-mean"] = gp_out_dict["mean"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-lower"] = gp_out_dict["lower"][:,-output_period_steps:]
        out_dict[f"gp_pred_future_{output_col}-upper"] = gp_out_dict["upper"][:,-output_period_steps:]

        """Pass the output of the gp into the gru"""
        gru_out_dict = self.model["gru"](**kwargs, **out_dict)

        out_dict.update(gru_out_dict)



        return out_dict

    def learning_step(self, **kwargs):
        out_dict = self(**kwargs)
        out_dict.update(kwargs)
        metrics = self.calculate_metrics(out_dict, optim_key="gru")

        self.optimizer["gru"].zero_grad()
        metrics["total_loss"].backward(retain_graph=True)
        self.optimizer["gru"].step()

        return out_dict,metrics


    def calculate_metrics(self, x, optim_key="gru"):
        """define you metrics that you will use"""

        score_dict = {"total_loss": 0}

        # lets start with the metrics and add them to the score dict
        for metric_key,metric_obj in self.metrics.items():
            # get the args from the x_args
            x_args = {k:x[v] for k,v in metric_obj["args"].items()}

            if "add_args" in metric_obj:
                x_args.update(metric_obj["add_args"])

            # get the loss from the function
            score_dict[metric_key] = metric_obj["fn"](**x_args)
            if "w" in metric_obj and (optim_key is not None and optim_key == metric_obj["optim"]):
                score_dict["total_loss"] += score_dict[metric_key] * metric_obj["w"]

        return score_dict


    def define_metrics(self):
        """calculate the metrics"""

        self.metrics = {}


        for output_name in self.model_args.predicted_columns:


            for mode in ["past", "future"]:
                self.metrics[f"MSE_gp_{mode}_{output_name}"]= \
                                                    dict(fn=partial(compute_mse, reduce_dims=[0,1]),
                                                        args=dict(y_pred=f"gp_pred_{mode}_{output_name}-mean",
                                                                  y_target=f"gt_{mode}_{output_name}"),
                                                        optim="gp")


                self.metrics[f"S1_{mode}_{output_name}"]= \
                                                    dict(fn=nn.SmoothL1Loss(),
                                                        args=dict(input=f"pred_{mode}_{output_name}-mean",
                                                                  target=f"gt_{mode}_{output_name}"),
                                                        w= self.model_args.loss_weight.mse,
                                                        optim="gru")

                self.metrics[f"MSE_{mode}_{output_name}"]= \
                                                    dict(fn=partial(compute_mse, reduce_dims=[0,1]),
                                                        args=dict(y_pred=f"pred_{mode}_{output_name}-mean",
                                                                  y_target=f"gt_{mode}_{output_name}"))

            """
            Map the get and pred to th 
            """
            for pair in self.args.data.transform.keys():
                output_transform_fn = self.args.data.transform[pair].column2transform[output_name]

                # pipeline for gt
                self.args.data.transform[pair].column2transform[f"gt_future_{output_name}"] = output_transform_fn
                self.args.data.transform[pair].column2transform[f"gt_past_{output_name}"] = output_transform_fn

                # pipeline for gp
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_future_{output_name}-lower"] = output_transform_fn

                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"gp_pred_past_{output_name}-lower"] =  output_transform_fn

                # pipeline for gp
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_future_{output_name}-lower"] = output_transform_fn


                self.args.data.transform[pair].column2transform[f"gt_past_{output_name}"] = output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-mean"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-upper"] =  output_transform_fn
                self.args.data.transform[pair].column2transform[f"pred_past_{output_name}-lower"] =  output_transform_fn