from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class InstanceNormalization:
    """
    Normalization of each window by the mean and standard deviation of its own past values (RevIN without the
    learnable affine transformation), which removes the level shifts between training and test windows.

    Attributes
    ----------
    epsilon : float
        Added to the variance before taking the square root.
    """

    epsilon: float = 1e-5

    def compute_statistics(self, past: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the location and scale of each window and channel.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, length, channels).

        Returns
        -------
        tuple[Tensor, Tensor]
            The mean and the standard deviation, each with shape (batch, 1, channels).
        """
        location: Tensor = past.mean(dim=1, keepdim=True).detach()
        scale: Tensor = torch.sqrt(past.var(dim=1, keepdim=True, unbiased=False) + self.epsilon).detach()
        return location, scale
