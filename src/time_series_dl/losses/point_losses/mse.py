from dataclasses import dataclass
from typing import override

from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class MSELoss(PointLoss):
    """
    Squared error, whose minimizer is the conditional mean.
    """

    @override
    def compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets).square()
