from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ContextualContrastiveLoss:
    """
    Contextual contrasting of TS-TCC, the NT-Xent loss of SimCLR: the projected contexts of the two views of a
    sequence are a positive pair, and the contexts of every other sequence of the batch are negatives.

    Attributes
    ----------
    temperature : float
        Temperature of the cosine similarities.

    Raises
    ------
    ValueError
        If temperature is not positive.
    """

    temperature: float = 0.2

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def compute(self, first: Tensor, second: Tensor) -> Tensor:
        """
        Compute the loss of two views.

        Parameters
        ----------
        first : Tensor
            Projected contexts of the first view with shape (batch, features).
        second : Tensor
            Projected contexts of the second view with the same shape.

        Returns
        -------
        Tensor
            The cross-entropy of choosing the partner view, averaged over the 2 * batch contexts.
        """
        count: int = first.shape[0]
        contexts: Tensor = functional.normalize(torch.cat([first, second], dim=0), dim=1)
        is_self: Tensor = torch.eye(2 * count, dtype=torch.bool, device=first.device)
        logits: Tensor = (contexts @ contexts.T / self.temperature).masked_fill(is_self, -torch.inf)
        items: Tensor = torch.arange(count, device=first.device)
        return functional.cross_entropy(logits, torch.cat([items + count, items]))
