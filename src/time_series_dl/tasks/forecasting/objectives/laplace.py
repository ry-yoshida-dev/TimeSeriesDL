from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.distributions import Laplace

from ..objective import ForecastObjective

_MINIMUM_SCALE: float = 1e-4


@dataclass(frozen=True)
class LaplaceObjective(ForecastObjective):
    """
    Laplace distribution of each forecast value, trained with the negative log-likelihood.

    The location is fitted like the MAE and the scale learns the uncertainty of each value. The parameters are the
    location and the scale; the point forecast is the location, which estimates the median.

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
        scale: Tensor = functional.softplus(raw[..., 1]) + _MINIMUM_SCALE
        return torch.stack([raw[..., 0], scale], dim=-1)

    @override
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        log_likelihood: Tensor = Laplace(parameters[..., 0], parameters[..., 1]).log_prob(targets)
        return -log_likelihood.mean()

    @override
    def compute_point(self, parameters: Tensor) -> Tensor:
        return parameters[..., 0]

    @override
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        levels: Tensor = torch.tensor(self.prediction_quantiles, device=parameters.device, dtype=parameters.dtype)
        distribution: Laplace = Laplace(parameters[..., 0:1], parameters[..., 1:2])
        quantiles: Tensor = distribution.icdf(levels)
        return quantiles

    @override
    def sample(self, parameters: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        uniform: Tensor = self.draw_uniform(parameters, num_samples, generator)
        samples: Tensor = Laplace(parameters[..., 0:1], parameters[..., 1:2]).icdf(uniform)
        return samples.permute(0, 3, 1, 2)
