from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn

from .irregular_tensors import IrregularTensors


class IrregularSequenceEncoder(nn.Module, ABC):
    """
    Maps irregularly sampled sequences, given by their observation times, values and observed masks, to one feature
    vector each, which the generic irregular heads turn into summaries.

    Time 0 is the common origin of every sample: tasks scale times so that the earliest training time is 0, and
    encoders may start from it so that absolute times stay informative.
    """

    @property
    @abstractmethod
    def output_size(self) -> int:
        """
        Get the size of the feature vector.

        Returns
        -------
        int
            The feature size.
        """

    @abstractmethod
    def forward(self, sequences: IrregularTensors) -> Tensor:
        """
        Encode a batch of sequences.

        Parameters
        ----------
        sequences : IrregularTensors
            The padded sequences.

        Returns
        -------
        Tensor
            Features with shape (batch, output_size), unchanged by padding steps.
        """

    def __call__(self, sequences: IrregularTensors) -> Tensor:
        return cast(Tensor, super().__call__(sequences))
