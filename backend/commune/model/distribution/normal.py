import torch
from torch import distributions
import torch.nn.functional as F
from .base import DistributionBase


class NormalDistribution(DistributionBase):
    """
    Normal distribution loss.

    Requirements for original target normalizer:
        * not normalized in log space (use :py:class:`~LogNormalDistributionLoss`)
        * not coerced to be positive
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        return self.distribution_class(loc=x[..., 0], scale=F.softplus(x[..., 1]))

class LogNormalDistribution(DistributionBase):
    """
    Log-normal loss.

    Requirements for original target normalizer:
        * normalized target in log space
    """
    distribution_class = distributions.LogNormal
    distribution_arguments = ["loc", "scale"]
    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.LogNormal:
        return self.distribution_class(loc=x[..., 0], scale=F.softplus(x[..., 1]))
    # def rescale_parameters(
    #     self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    # ) -> torch.Tensor:
    #     assert isinstance(encoder.block, str) and encoder.block in [
    #         "log",
    #         "log1p",
    #     ], f"Log distribution requires log scaling but found `block={encoder.transform}`"
    #
    #     assert encoder.block not in ["logit"], "Cannot use bound block such as 'logit'"
    #
    #     scale = F.softplus(parameters[..., 1]) * target_scale[..., 1].unsqueeze(-1)
    #     loc = parameters[..., 0] * target_scale[..., 1].unsqueeze(-1) + target_scale[..., 0].unsqueeze(-1)
    #
    #     return torch.stack([loc, scale], dim=-1)