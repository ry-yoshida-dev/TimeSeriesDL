from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..objective import ForecastObjective


@dataclass(frozen=True)
class QuantileObjective(ForecastObjective):
    """
    Predicted quantiles of each forecast value, trained with the pinball loss.

    Predicted quantiles are sorted, so they never cross. The point forecast is the quantile closest to the median.

    Attributes
    ----------
    quantiles : tuple[float, ...]
        Levels of the predicted quantiles, strictly increasing in (0, 1).

    Raises
    ------
    ValueError
        If the levels are empty, out of (0, 1) or not strictly increasing.
    """

    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def __post_init__(self) -> None:
        if not self.quantiles:
            raise ValueError("quantiles must not be empty.")
        if any(not 0.0 < level < 1.0 for level in self.quantiles):
            raise ValueError(f"quantiles must be in (0, 1), got {self.quantiles}")
        if any(lower >= upper for lower, upper in zip(self.quantiles, self.quantiles[1:], strict=False)):
            raise ValueError(f"quantiles must be strictly increasing, got {self.quantiles}")

    @property
    @override
    def num_parameters(self) -> int:
        return len(self.quantiles)

    @property
    @override
    def location_index(self) -> int:
        distances: list[float] = [abs(level - 0.5) for level in self.quantiles]
        return distances.index(min(distances))

    @property
    @override
    def level_parameters(self) -> tuple[int, ...]:
        return tuple(range(len(self.quantiles)))

    @property
    @override
    def scale_parameters(self) -> tuple[int, ...]:
        return ()

    @property
    @override
    def quantile_levels(self) -> tuple[float, ...]:
        return self.quantiles

    @override
    def constrain(self, raw: Tensor) -> Tensor:
        return torch.sort(raw, dim=-1).values

    @override
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        levels: Tensor = torch.tensor(self.quantiles, device=parameters.device, dtype=parameters.dtype)
        errors: Tensor = targets.unsqueeze(-1) - parameters
        return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()

    @override
    def compute_point(self, parameters: Tensor) -> Tensor:
        return parameters[..., self.location_index]

    @override
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        return parameters

    @override
    def sample(self, parameters: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        uniform: Tensor = self.draw_uniform(parameters, num_samples, generator)
        count: int = len(self.quantiles)
        if count == 1:
            return parameters.expand(*parameters.shape[:-1], num_samples).permute(0, 3, 1, 2)
        levels: Tensor = torch.tensor(self.quantiles, device=parameters.device, dtype=parameters.dtype)
        upper: Tensor = torch.searchsorted(levels, uniform.contiguous()).clamp(1, count - 1)
        lower: Tensor = upper - 1
        weight: Tensor = ((uniform - levels[lower]) / (levels[upper] - levels[lower])).clamp(0.0, 1.0)
        lower_values: Tensor = parameters.gather(-1, lower)
        upper_values: Tensor = parameters.gather(-1, upper)
        return (lower_values + weight * (upper_values - lower_values)).permute(0, 3, 1, 2)
