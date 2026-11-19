"""Structural type for models that build reconstruction networks."""

from typing import Protocol, runtime_checkable

from ..reconstruction_network import ReconstructionNetwork
from ..shapes import SequenceShape


@runtime_checkable
class ReconstructionModel(Protocol):
    """
    Model parameters able to build a reconstruction network, for imputation and anomaly detection.
    """

    def build_reconstruction_network(self, shape: SequenceShape) -> ReconstructionNetwork:
        """
        Build an untrained reconstruction network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the input and output sequences.

        Returns
        -------
        ReconstructionNetwork
            The network.
        """
        ...
