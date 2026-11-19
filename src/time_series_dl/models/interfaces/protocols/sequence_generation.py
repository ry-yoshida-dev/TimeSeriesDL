"""Structural type for models that generate whole sequences."""

from typing import Protocol, runtime_checkable

from ..sequence_generation_network import SequenceGenerationNetwork
from ..shapes import SequenceShape


@runtime_checkable
class SequenceGenerationModel(Protocol):
    """
    Model parameters able to build a sequence generation network, for synthetic time series generation.
    """

    def build_sequence_generation_network(self, shape: SequenceShape) -> SequenceGenerationNetwork:
        """
        Build an untrained sequence generation network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences it generates.

        Returns
        -------
        SequenceGenerationNetwork
            The network.
        """
        ...
