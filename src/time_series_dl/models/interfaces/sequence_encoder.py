from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn


class SequenceEncoder(nn.Module, ABC):
    """
    Maps a sequence to a sequence of feature vectors, which the generic heads of ``components.heads`` turn into
    forecasts, reconstructions or summaries.
    """

    @property
    @abstractmethod
    def output_length(self) -> int:
        """
        Get the number of feature vectors per sequence.

        Returns
        -------
        int
            The output length.
        """

    @property
    @abstractmethod
    def output_size(self) -> int:
        """
        Get the size of each feature vector.

        Returns
        -------
        int
            The feature size.
        """

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """
        Encode a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Features with shape (batch, output_length, output_size).
        """

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
