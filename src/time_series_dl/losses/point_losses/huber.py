from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class HuberLoss(PointLoss):
    """
    Huber loss: quadratic for errors up to ``delta`` and linear beyond, so large errors weigh like MAE.

    Attributes
    ----------
    delta : float
        Error at which the loss turns from quadratic to linear, in the scale of the targets the loss sees.

    Raises
    ------
    ValueError
        If delta is not positive.
    """

    delta: float = 1.0

    def __post_init__(self) -> None:
        if self.delta <= 0.0:
            raise ValueError(f"delta must be positive, got {self.delta}")

    @override
    def compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return functional.huber_loss(predictions, targets, reduction="none", delta=self.delta)
