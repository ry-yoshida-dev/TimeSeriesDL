import math
from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..point_loss import PointLoss


@dataclass(frozen=True)
class LogCoshLoss(PointLoss):
    """
    Logarithm of the hyperbolic cosine of the error: about half the squared error near zero and the absolute error
    beyond, smooth everywhere.
    """

    @override
    def compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2), which does not overflow for large errors.
        errors: Tensor = (predictions - targets).abs()
        return errors + functional.softplus(-2.0 * errors) - math.log(2.0)
