"""Structural type for models that build irregular interpolation networks."""

from typing import Protocol, runtime_checkable

from ..irregular_interpolation_network import IrregularInterpolationNetwork
from ..shapes.irregular_sequence import IrregularSequenceShape


@runtime_checkable
class IrregularInterpolationModel(Protocol):
    """
    Model parameters able to build an irregular interpolation network, for interpolation and forecasting of
    irregularly sampled sequences at query times.
    """

    def build_irregular_interpolation_network(self, shape: IrregularSequenceShape) -> IrregularInterpolationNetwork:
        """
        Build an untrained irregular interpolation network.

        Parameters
        ----------
        shape : IrregularSequenceShape
            The dimensions of the sequences.

        Returns
        -------
        IrregularInterpolationNetwork
            The network.
        """
        ...
