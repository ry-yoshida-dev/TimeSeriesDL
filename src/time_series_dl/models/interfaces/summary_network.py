from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn


class SummaryNetwork(nn.Module, ABC):
    """
    Maps a whole sequence to one output vector, such as class logits or regression targets.
    """

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """
        Summarize a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Outputs with shape (batch, output_size).
        """

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
