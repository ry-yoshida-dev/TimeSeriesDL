from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Masking(ABC):
    """
    Random choice of the values that masked modeling hides from an encoder.
    """

    @abstractmethod
    def sample(self, batch: int, length: int, channels: int, device: torch.device) -> Tensor:
        """
        Draw a new mask.

        Parameters
        ----------
        batch : int
            Number of sequences.
        length : int
            Number of time steps of each sequence.
        channels : int
            Number of channels of each sequence.
        device : torch.device
            Device of the mask.

        Returns
        -------
        Tensor
            Booleans with shape (batch, length, channels), True for each hidden value.
        """
