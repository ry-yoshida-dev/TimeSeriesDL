from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class MAELoss(PointLoss):
    """
    Absolute error, whose minimizer is the conditional median; robust to outliers.
    """

    @override
    def compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets).abs()
