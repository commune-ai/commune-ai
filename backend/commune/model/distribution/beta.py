import torch
from torch import distributions
from .base import DistributionBase


class BetaDistribution(DistributionBase):
    """
    Beta distribution loss for unit interval data.

    Requirements for original target normalizer:
        * logit block
    """

    distribution_class = distributions.Beta
    distribution_arguments = ["mean", "shape"]
    eps = 1e-4

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Beta:
        mean = x[..., 0]
        shape = x[..., 1]
        return self.distribution_class(concentration0=(1 - mean) * shape, concentration1=mean * shape)

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        # clip y_actual to avoid infinite losses
        loss = -self.distribution.log_prob(y_actual.clip(self.eps, 1 - self.eps))
        return loss

    # def rescale_parameters(
    #     self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    # ) -> torch.Tensor:
    #     assert encoder.block in ["logit"], "Beta distribution is only compatible with logit block"
    #     assert encoder.center, "Beta distribution requires normalizer to center data"
    #
    #     scaled_mean = encoder(dict(prediction=parameters[..., 0], target_scale=target_scale))
    #     # need to first transform target scale standard deviation in logit space to real space
    #     # we assume a normal distribution in logit space (we used a logit transform and a standard scaler)
    #     # and know that the variance of the beta distribution is limited by `scaled_mean * (1 - scaled_mean)`
    #     scaled_mean = scaled_mean * (1 - 2 * self.eps) + self.eps  # ensure that mean is not exactly 0 or 1
    #     mean_derivative = scaled_mean * (1 - scaled_mean)
    #
    #     # we can approximate variance as
    #     # torch.pow(torch.tanh(target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)), 2) * mean_derivative
    #     # shape is (positive) parameter * mean_derivative / var
    #     shape_scaler = (
    #         torch.pow(torch.tanh(target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)), 2) + self.eps
    #     )
    #     scaled_shape = F.softplus(parameters[..., 1]) / shape_scaler
    #     return torch.stack([scaled_mean, scaled_shape], dim=-1)