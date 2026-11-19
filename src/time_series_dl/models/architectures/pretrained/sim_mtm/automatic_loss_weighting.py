from typing import cast

import torch
from torch import Tensor, nn


class SimMTMAutomaticLossWeighting(nn.Module):
    """
    Learned weighting of several losses used by the official SimMTM code (Liebel and Koerner, 2018): each loss
    ``L_i`` contributes ``0.5 / w_i^2 * L_i + log(1 + w_i^2)`` with a trained ``w_i`` that starts at 1.

    Parameters
    ----------
    num_losses : int
        Number of weighted losses.
    """

    def __init__(self, num_losses: int) -> None:
        super().__init__()
        self.weights: nn.Parameter = nn.Parameter(torch.ones(num_losses))

    def forward(self, losses: Tensor) -> Tensor:
        """
        Combine the losses.

        Parameters
        ----------
        losses : Tensor
            The scalar losses stacked with shape (num_losses,).

        Returns
        -------
        Tensor
            The scalar weighted sum.
        """
        squared: Tensor = self.weights.square()
        return (0.5 / squared * losses + torch.log1p(squared)).sum()

    def __call__(self, losses: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(losses))
