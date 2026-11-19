from dataclasses import dataclass, field
from typing import override

import torch
from deep_learning_loss import MSELoss, PointLoss
from torch import Tensor

from ..objective import ForecastObjective


@dataclass(frozen=True)
class PointObjective(ForecastObjective):
    """
    One predicted value per forecast value, trained with a point loss.

    Attributes
    ----------
    loss : PointLoss
        The training loss on the scaled values.
    """

    loss: PointLoss = field(default_factory=MSELoss)

    @property
    @override
    def num_parameters(self) -> int:
        return 1

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
        return ()

    @property
    @override
    def quantile_levels(self) -> tuple[float, ...]:
        return ()

    @override
    def constrain(self, raw: Tensor) -> Tensor:
        return raw

    @override
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        return self.loss.compute(parameters[..., 0], targets)

    @override
    def compute_point(self, parameters: Tensor) -> Tensor:
        return parameters[..., 0]

    @override
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        return parameters[..., :0]

    @override
    def sample(self, parameters: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        return parameters[..., 0].unsqueeze(1).expand(-1, num_samples, -1, -1)
