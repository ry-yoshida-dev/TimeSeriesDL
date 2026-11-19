from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn


class ReconstructionNetwork(nn.Module, ABC):
    """
    Maps a sequence to a sequence of the same shape, reconstructing hidden or corrupted values.
    """

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """
        Reconstruct a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels); missing values are zero.

        Returns
        -------
        Tensor
            Reconstructed values with shape (batch, length, channels).
        """

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
