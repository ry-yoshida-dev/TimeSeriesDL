from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn


class StateSpaceLayer(nn.Module, ABC):
    """
    Maps a sequence of feature vectors to a sequence of the same size through a state space model, including the
    layer's own output projection; the residual connection and normalization belong to the encoder.
    """

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the layer causally along time.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, size).

        Returns
        -------
        Tensor
            Outputs with shape (batch, length, size).
        """

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
