"""Structural type for models that build mask-aware reconstruction networks."""

from typing import Protocol, runtime_checkable

from ..masked_reconstruction_network import MaskedReconstructionNetwork
from ..shapes import SequenceShape


@runtime_checkable
class MaskedReconstructionModel(Protocol):
    """
    Model parameters able to build a reconstruction network that reads the observed mask, for imputation.
    """

    def build_masked_reconstruction_network(self, shape: SequenceShape) -> MaskedReconstructionNetwork:
        """
        Build an untrained masked reconstruction network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the input and output sequences.

        Returns
        -------
        MaskedReconstructionNetwork
            The network.
        """
        ...
