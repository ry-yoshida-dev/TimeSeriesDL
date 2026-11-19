"""Structural type for models that build irregular summary networks."""

from typing import Protocol, runtime_checkable

from ..irregular_summary_network import IrregularSummaryNetwork
from ..shapes.irregular_summary import IrregularSummaryShape


@runtime_checkable
class IrregularSummaryModel(Protocol):
    """
    Model parameters able to build an irregular summary network, for classification and regression of irregularly
    sampled sequences.
    """

    def build_irregular_summary_network(self, shape: IrregularSummaryShape) -> IrregularSummaryNetwork:
        """
        Build an untrained irregular summary network.

        Parameters
        ----------
        shape : IrregularSummaryShape
            The input and output dimensions.

        Returns
        -------
        IrregularSummaryNetwork
            The network.
        """
        ...
