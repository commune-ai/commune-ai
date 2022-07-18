import torch
from torch import distributions
from typing import Any, Dict, List, Tuple, Union


class DistributionBase():
    """
    DistributionLoss base class.

    Class should be inherited for all distribution losses, i.e. if a network predicts
    the parameters of a probability distribution, DistributionLoss can be used to
    score those parameters and calculate loss for given true values.

    Define two class attributes in a child class:

    Attributes:
        distribution_class (distributions.Distribution): torch probability distribution
        distribution_arguments (List[str]): list of parameter names for the distribution

    Further, implement the methods :py:meth:`~map_x_to_distribution` and :py:meth:`~rescale_parameters`.
    """

    distribution_class: distributions.Distribution
    distribution_arguments: List[str]
    distribution: distributions.Distribution = None

    def __init__(self, x: torch.Tensor):
        self.device = x.device
        self.distribution = self.map_x_to_distribution(x)

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
        """
        Map the a tensor of parameters to a probability distribution.

        Args:
            x (torch.Tensor): parameters for probability distribution. Last dimension will index the parameters

        Returns:
            distributions.Distribution: torch probability distribution as defined in the
                class attribute ``distribution_class``
        """
        raise NotImplementedError("implement this method")

    def loss(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """

        loss = -self.distribution.log_prob(y)
        return loss

    def to_prediction(self,) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: mean prediction
        """

        return self.distribution.mean

    def sample(self, n_samples: int=1) -> torch.Tensor:
        """
        Sample from distribution.

        Args:
            y_pred: prediction output of network (shape batch_size x n_timesteps x n_paramters)
            n_samples (int): number of samples to draw

        Returns:
            torch.Tensor: tensor with samples  (shape batch_size x n_timesteps x n_samples)
        """
        samples = self.distribution.sample((n_samples,))

        return samples

    def to_quantiles(self,
                     quantiles: List[float] = None,
                     n_samples: int = 100) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.
            n_samples (int): number of samples to draw for quantiles. Defaults to 100.

        Returns:
            torch.Tensor: prediction quantiles (last dimension)
        """
        try:
            quantiles = self.distribution.icdf(torch.tensor(quantiles, device=self.device)[:, None, None]).permute(1, 2, 0)
        except NotImplementedError:  # resort to derive quantiles empirically
            samples = torch.sort(self.sample(n_samples), -1).values
            quantiles = torch.quantile(samples, torch.tensor(quantiles, device=self.device), dim=2).permute(1, 2, 0)
        return quantiles

