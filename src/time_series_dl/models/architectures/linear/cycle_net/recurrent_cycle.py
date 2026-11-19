from typing import cast

import torch
from torch import Tensor, nn


class CycleNetRecurrentCycle(nn.Module):
    """
    Learnable recurrent cycle of CycleNet: one value per phase of the cycle and channel, initialized to zero, read
    at the phase of each time step.

    Parameters
    ----------
    cycle_length : int
        Number of phases of the cycle.
    channels : int
        Number of channels.
    """

    def __init__(self, cycle_length: int, channels: int) -> None:
        super().__init__()
        self.cycle: nn.Parameter = nn.Parameter(torch.zeros(cycle_length, channels))

    def forward(self, phase: Tensor) -> Tensor:
        """
        Read the cycle at the given phases.

        Parameters
        ----------
        phase : Tensor
            Integer phases in ``[0, cycle_length)`` with shape (batch, length).

        Returns
        -------
        Tensor
            The cycle values with shape (batch, length, channels).
        """
        return self.cycle[phase]

    def __call__(self, phase: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(phase))
