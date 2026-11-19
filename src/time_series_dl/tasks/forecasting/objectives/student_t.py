from dataclasses import dataclass
from typing import override

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.special import stdtrit
from torch import Tensor
from torch.distributions import StudentT

from ....array_types import FloatArray
from ..objective import ForecastObjective

_MINIMUM_SCALE: float = 1e-4
_MINIMUM_DEGREES_OF_FREEDOM: float = 2.0


@dataclass(frozen=True)
class StudentTObjective(ForecastObjective):
    """
    Student's t distribution of each forecast value, trained with the negative log-likelihood, as in DeepAR.

    Its heavy tails make it robust to outliers and spikes that a Gaussian would overreact to. The parameters are the
    location, the scale and the degrees of freedom, kept above 2 so the variance is finite; the point forecast is
    the location.

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
        return 3

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
        degrees_of_freedom: Tensor = functional.softplus(raw[..., 2]) + _MINIMUM_DEGREES_OF_FREEDOM
        return torch.stack([raw[..., 0], scale, degrees_of_freedom], dim=-1)

    @override
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        distribution: StudentT = StudentT(parameters[..., 2], parameters[..., 0], parameters[..., 1])
        log_likelihood: Tensor = distribution.log_prob(targets)
        return -log_likelihood.mean()

    @override
    def compute_point(self, parameters: Tensor) -> Tensor:
        return parameters[..., 0]

    @override
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        # PyTorch has no inverse CDF of Student's t; quantiles are only needed for predictions, outside autograd.
        degrees_of_freedom: FloatArray = parameters[..., 2:3].detach().double().cpu().numpy()
        levels: FloatArray = np.asarray(self.prediction_quantiles, dtype=np.float64)
        standard_quantiles: FloatArray = np.asarray(stdtrit(degrees_of_freedom, levels), dtype=np.float64)
        standard: Tensor = torch.from_numpy(standard_quantiles).to(parameters)
        return parameters[..., 0:1] + parameters[..., 1:2] * standard
