"""Structural type for models that build summary networks."""

from typing import Protocol, runtime_checkable

from ..shapes import SummaryShape
from ..summary_network import SummaryNetwork


@runtime_checkable
class SummaryModel(Protocol):
    """
    Model parameters able to build a summary network, for classification and regression.
    """

    def build_summary_network(self, shape: SummaryShape) -> SummaryNetwork:
        """
        Build an untrained summary network.

        Parameters
        ----------
        shape : SummaryShape
            The input and output dimensions.

        Returns
        -------
        SummaryNetwork
            The network.
        """
        ...
