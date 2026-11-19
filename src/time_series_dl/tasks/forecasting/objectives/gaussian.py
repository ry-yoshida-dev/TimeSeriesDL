from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.distributions import Normal

from ..objective import ForecastObjective

_MINIMUM_STANDARD_DEVIATION: float = 1e-4


@dataclass(frozen=True)
class GaussianObjective(ForecastObjective):
    """
    Normal distribution of each forecast value, trained with the negative log-likelihood, as in DeepAR.

    The parameters are the mean and the standard deviation; the point forecast is the mean.

    Attributes
    ----------
    prediction_quantiles : tuple[float, ...]
        Levels of the quantiles that predictions report, strictly increasing in (0, 1).

    Raises
    ------
    ValueError
        If the levels are out of (0, 1) or not strictly increasing.
    """

    prediction_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
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
        return 2

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
        return (1,)

    @property
    @override
    def quantile_levels(self) -> tuple[float, ...]:
        return self.prediction_quantiles

    @override
    def constrain(self, raw: Tensor) -> Tensor:
        deviation: Tensor = functional.softplus(raw[..., 1]) + _MINIMUM_STANDARD_DEVIATION
        return torch.stack([raw[..., 0], deviation], dim=-1)

    @override
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        return functional.gaussian_nll_loss(parameters[..., 0], targets, parameters[..., 1].square(), full=True)

    @override
    def compute_point(self, parameters: Tensor) -> Tensor:
        return parameters[..., 0]

    @override
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        levels: Tensor = torch.tensor(self.prediction_quantiles, device=parameters.device, dtype=parameters.dtype)
        distribution: Normal = Normal(parameters[..., 0:1], parameters[..., 1:2])
        quantiles: Tensor = distribution.icdf(levels)
        return quantiles

    @override
    def sample(self, parameters: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        normal: Tensor = self.draw_normal(parameters, num_samples, generator)
        return (parameters[..., 0:1] + parameters[..., 1:2] * normal).permute(0, 3, 1, 2)
