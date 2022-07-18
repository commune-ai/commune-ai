import torch
from torch import distributions
from .base import DistributionBase

class NegativeBinomialDistribution(DistributionBase):
    """
    Negative binomial loss, e.g. for count data.

    Requirements for original target normalizer:
        * not centered normalization (only rescaled)
    """

    distribution_class = distributions.NegativeBinomial
    distribution_arguments = ["mean", "shape"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.NegativeBinomial:
        mean = x[..., 0]
        shape = x[..., 1]
        r = 1.0 / shape
        p = mean / (mean + r)
        return self.distribution_class(total_count=r, probs=p)

    # def rescale_parameters(
    #     self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    # ) -> torch.Tensor:
    #     assert not encoder.center, "NegativeBinomialDistributionLoss is not compatible with `center=True` normalization"
    #     assert encoder.block not in ["logit"], "Cannot use bound block such as 'logit'"
    #     if encoder.block in ["log", "log1p"]:
    #         mean = torch.exp(parameters[..., 0] * target_scale[..., 1].unsqueeze(-1))
    #         shape = (
    #             F.softplus(torch.exp(parameters[..., 1]))
    #             / torch.exp(target_scale[..., 1].unsqueeze(-1)).sqrt()  # todo: is this correct?
    #         )
    #     else:
    #         mean = F.softplus(parameters[..., 0]) * target_scale[..., 1].unsqueeze(-1)
    #         shape = F.softplus(parameters[..., 1]) / target_scale[..., 1].unsqueeze(-1).sqrt()
    #     return torch.stack([mean, shape], dim=-1)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction. In the case of this distribution prediction we
        need to derive the mean (as a point prediction) from the distribution parameters

        Args:
            y_pred: prediction output of network
            in this case the two parameters for the negative binomial

        Returns:
            torch.Tensor: mean prediction
        """
        return y_pred[..., 0]