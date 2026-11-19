"""Structural type for models that build self-supervised pretraining networks."""

from typing import Protocol, runtime_checkable

from ..pretraining_network import PretrainingNetwork
from ..shapes import SequenceShape


@runtime_checkable
class PretrainingModel(Protocol):
    """
    Model parameters able to build a self-supervised pretraining network, for representation learning.
    """

    def build_pretraining_network(self, shape: SequenceShape) -> PretrainingNetwork:
        """
        Build an untrained pretraining network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the unlabeled sequences.

        Returns
        -------
        PretrainingNetwork
            The network.
        """
        ...
