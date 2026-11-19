from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class PinballLoss(PointLoss):
    """
    Pinball (quantile) loss, whose minimizer is the conditional quantile of ``level``.

    Under-predictions weigh ``level`` and over-predictions ``1 - level``; level 0.5 gives half the absolute error.

    Attributes
    ----------
    level : float
        The quantile level, in (0, 1).

    Raises
    ------
    ValueError
        If level is not in (0, 1).
    """

    level: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level}")

    @override
    def compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        errors: Tensor = targets - predictions
        return torch.maximum(self.level * errors, (self.level - 1.0) * errors)
