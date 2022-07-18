
import os
from functools import partial
import numpy as np
import torch
from torch import nn
from typing import Dict, List, Tuple

# GPyTorch Imports
import gpytorch
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, MultitaskKernel
from gpytorch.kernels import RBFKernel, RBFKernel, ProductKernel
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList, MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
EPS = 1E-10

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x=None, train_y=None, likelihood=None, batch_size=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([batch_size])),
            batch_shape=torch.Size([batch_size])
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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

        x = torch.linspace(-1, 1, full_period_steps)\
                        .unsqueeze(0)\
                        .repeat(batch_size,1)\
                        .unsqueeze(2).to(self.args.trainer.device)

        x_rand_idx = torch.randperm(x.shape[1]).to(self.args.trainer.device)
        x = torch.index_select(x, 1, x_rand_idx )



        input_period_steps = int(full_period_steps*sample_freq)

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
    def forward(self, y, sample_freq=1.0, extension_periods=[0, 0], preserve_endpoints=[0.1, 0.1]):
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



        idx = torch.arange(0, x.shape[1]).to(device)

        preserve_endpoints = [int(preserve_endpoints[0]*x.shape[1]),
                              x.shape[1]-int(preserve_endpoints[1] * x.shape[1]) ]
        preserved_endpoint_idx = torch.cat([idx[:preserve_endpoints[0]],
                                            idx[preserve_endpoints[1]:]])

        sample_idx = idx[preserve_endpoints[0]:preserve_endpoints[1]]
        sample_idx = torch.randperm(sample_idx.shape[0]).to(device)[:int(sample_freq*sample_idx.shape[0])]

        sample_idx = torch.cat([preserved_endpoint_idx, sample_idx])

        x_train = torch.index_select(x, 1, sample_idx ).to(device)
        y_train = torch.index_select(y, 1, sample_idx ).to(device)

        self.gp.set_train_data(inputs=x_train, targets=y_train, strict=False)
        # fit
        self.fit(x=x_train, y=y_train)

        x_extend = torch.arange(min_x - extension_periods[0]*step_size,
                                max_x +  extension_periods[1]*step_size,
                                step_size).to(device)

        y_extend= self.inference(x_extend)["mean"]

        y_extend  = y_extend*(y_std+EPS) + y_mean

        return y_extend




class MultivariateBatchedGP(ExactGP):
    """Class for creating batched Gaussian Process Regression models.  Ideal candidate if
    using GPU-based acceleration such as CUDA for training.
    This kernel produces a compose kernel that multiplies actions times states,
    i.e. we have a different kernel for both the actions and states.  In turn,
    the compose kernel is then multiplied by a Scale kernel.
    Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (B * YD, N, XD), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) XD is the dimension of the features (d_state + d_action)
                (iv) YD is the dimension of the labels (d_reward + d_state)
            The features of train_x are tiled YD times along the first dimension.
        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (B * YD, N), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) YD is the dimension of the labels (d_reward + d_state)
            The features of train_y are stacked.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.
        shape (int):  The batch shape used for creating this BatchedGP model.
            This corresponds to the number of samples we wish to interpolate.
        output_device (str):  The device on which the GPR will be trained on.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        ds (int): If using a compose kernel, ds specifies the dimensionality of
            the state.  Only applicable if compose_kernel is True.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim:int,
                 batch_size:int,
                 device: str,
                 periods: Dict,
                 use_ard = False,
                 ):



        self.__dict__.update(locals())


        # Check if ds is None, and if not, set
        # Set active dimensions
        self.active_dims = torch.tensor([i for i in range(0, input_dim)], device=device)
        # Determine if using ARD

        # Create the mean and covariance modules
        self.ard_num_dims = self.input_dim if self.use_ard else None

        self.create_model(batch_size=batch_size)

    def create_model(self, batch_size ):


        self.batch_size = batch_size*self.output_dim

        self.batch_shape = torch.Size([self.batch_size])

        dummy_train_input = torch.rand((self.batch_size, self.periods['input'], self.input_dim), device=self.device)
        dummy_train_target = torch.rand((self.batch_size, self.periods['input']), device=self.device)

        # Run constructor of superclass
        super(MultivariateBatchedGP, self).__init__(train_inputs=dummy_train_input,
                                                    train_targets=dummy_train_target,
                                                    likelihood=GaussianLikelihood(batch_shape=self.batch_shape))



        # Construct mean module
        self.mean_module = ConstantMean(batch_shape=self.batch_shape,
                                        output_device=self.device)

        # Construct state kernel
        self.base_kernel = RBFKernel(batch_shape=self.batch_shape,
                                              active_dims=self.active_dims,
                                              ard_num_dims=self.ard_num_dims)

        self.covar_module = ScaleKernel(self.base_kernel,
                                        batch_shape=self.batch_shape,
                                        output_device=self.device)
        self.original_model_state = self.state_dict()

    def reset_model_state(self):
        self.load_state_dict(self.original_model_state)

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.
        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.
        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        # Compute mean and covariance in batched form
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)