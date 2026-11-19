import math
from dataclasses import dataclass
from typing import cast, override

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.distributions import Normal

from ..objective import ForecastObjective

_MINIMUM_STANDARD_DEVIATION: float = 1e-4


@dataclass(frozen=True)
class LowRankGaussianObjective(ForecastObjective):
    """
    Multivariate normal distribution of the target channels at each horizon step, with a low-rank plus diagonal
    covariance, trained with the negative log-likelihood, as in GPVar (Salinas et al., 2019).

    The covariance of the channels at one step is ``diag(d^2) + V V^T``, where channel ``i`` has the mean ``mu_i``,
    the standard deviation ``d_i`` and the factor row ``v_i`` of length ``rank``; its parameters are
    ``(mu_i, d_i, v_i1, ..., v_irank)``, so every network that outputs per-value parameters can predict it, and the
    factors of different channels couple them. Horizon steps are independent given the parameters. Since a channel's
    scale multiplies its mean, deviation and factor row alike, instance normalization rescales the distribution
    exactly. The loss is the negative log-likelihood per channel; the point forecast is the mean, and quantiles are
    those of each channel's marginal normal distribution.

    Attributes
    ----------
    rank : int
        Rank of the low-rank part of the covariance.
    prediction_quantiles : tuple[float, ...]
        Levels of the quantiles that predictions report, strictly increasing in (0, 1).

    Raises
    ------
    ValueError
        If rank is not positive, or the levels are out of (0, 1) or not strictly increasing.
    """

    rank: int = 10
    prediction_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if any(not 0.0 < level < 1.0 for level in self.prediction_quantiles):
            raise ValueError(f"prediction_quantiles must be in (0, 1), got {self.prediction_quantiles}")
        if any(
            lower >= upper
            for lower, upper in zip(self.prediction_quantiles, self.prediction_quantiles[1:], strict=False)
        ):
            raise ValueError(f"prediction_quantiles must be strictly increasing, got {self.prediction_quantiles}")

    @property
    @override
    def num_parameters(self) -> int:
        return self.rank + 2

    @property
    @override
    def location_index(self) -> int:
        return 0

    @property
    @override
    def level_parameters(self) -> tuple[int, ...]:
        return (0,)

    @property
    @override
    def scale_parameters(self) -> tuple[int, ...]:
        return tuple(range(1, self.rank + 2))

    @property
    @override
    def quantile_levels(self) -> tuple[float, ...]:
        return self.prediction_quantiles

    @override
    def constrain(self, raw: Tensor) -> Tensor:
        deviation: Tensor = functional.softplus(raw[..., 1:2]) + _MINIMUM_STANDARD_DEVIATION
        return torch.cat([raw[..., 0:1], deviation, raw[..., 2:]], dim=-1)

    def compute_log_likelihood(self, parameters: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the log-likelihood of the target channels at each step.

        The Woodbury identity and the matrix determinant lemma reduce the work to a ``rank`` x ``rank`` Cholesky
        factorization per step.

        Parameters
        ----------
        parameters : Tensor
            Distribution parameters with shape (batch, horizon, channels, rank + 2).
        targets : Tensor
            True values with shape (batch, horizon, channels).

        Returns
        -------
        Tensor
            The log-likelihood of each window and step with shape (batch, horizon).
        """
        variance: Tensor = parameters[..., 1].square()
        factor: Tensor = parameters[..., 2:]
        residual: Tensor = targets - parameters[..., 0]
        scaled_factor: Tensor = factor / variance.unsqueeze(-1)
        identity: Tensor = torch.eye(self.rank, device=parameters.device, dtype=parameters.dtype)
        capacitance: Tensor = identity + torch.einsum("bhcr,bhcs->bhrs", factor, scaled_factor)
        cholesky: Tensor = cast(Tensor, torch.linalg.cholesky(capacitance))
        projected: Tensor = torch.einsum("bhcr,bhc->bhr", scaled_factor, residual)
        solved: Tensor = cast(
            Tensor, torch.linalg.solve_triangular(cholesky, projected.unsqueeze(-1), upper=False)
        ).squeeze(-1)
        mahalanobis: Tensor = (residual.square() / variance).sum(dim=-1) - solved.square().sum(dim=-1)
        log_determinant: Tensor = 2.0 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1) + variance.log().sum(
            dim=-1
        )
        channels: int = targets.shape[-1]
        return -0.5 * (channels * math.log(2.0 * math.pi) + log_determinant + mahalanobis)

    @override
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        return -self.compute_log_likelihood(parameters, targets).mean() / targets.shape[-1]

    @override
    def compute_point(self, parameters: Tensor) -> Tensor:
        return parameters[..., 0]

    @override
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        levels: Tensor = torch.tensor(self.prediction_quantiles, device=parameters.device, dtype=parameters.dtype)
        deviation: Tensor = (parameters[..., 1:2].square() + parameters[..., 2:].square().sum(-1, keepdim=True)).sqrt()
        quantiles: Tensor = Normal(parameters[..., 0:1], deviation).icdf(levels)
        return quantiles

    @override
    def sample(self, parameters: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        independent: Tensor = self.draw_normal(parameters, num_samples, generator)
        shared: Tensor = torch.randn(
            (parameters.shape[0], parameters.shape[1], self.rank, num_samples),
            generator=generator,
            device=parameters.device,
            dtype=parameters.dtype,
        )
        correlated: Tensor = torch.einsum("bhcr,bhrn->bhcn", parameters[..., 2:], shared)
        samples: Tensor = parameters[..., 0:1] + parameters[..., 1:2] * independent + correlated
        return samples.permute(0, 3, 1, 2)
