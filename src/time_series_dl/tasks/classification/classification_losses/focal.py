from dataclasses import dataclass
from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..classification_loss import ClassificationLoss


@dataclass(frozen=True)
class FocalLoss(ClassificationLoss):
    """
    Focal loss (Lin et al., 2017): cross-entropy scaled by ``(1 - p) ** gamma``, where ``p`` is the predicted
    probability of the true class, so well-classified sequences weigh less and hard ones dominate.

    Attributes
    ----------
    gamma : float
        Focusing strength; 0 gives the cross-entropy.

    Raises
    ------
    ValueError
        If gamma is negative.
    """

    gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")

    @override
    def compute(self, logits: Tensor, indices: Tensor, class_weights: Tensor | None) -> Tensor:
        log_probabilities: Tensor = functional.log_softmax(logits, dim=1).gather(1, indices.unsqueeze(1)).squeeze(1)
        losses: Tensor = -((1.0 - log_probabilities.exp()) ** self.gamma) * log_probabilities
        if class_weights is None:
            return losses.mean()
        weights: Tensor = class_weights[indices]
        return (weights * losses).sum() / weights.sum()
