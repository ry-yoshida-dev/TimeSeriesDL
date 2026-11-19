"""Structural type for models whose networks can start from published pretrained weights."""

from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class PublishedWeightsModel(Protocol):
    """
    Model parameters whose networks can start from officially published weights, which the builders download (or
    read from the local cache) and load when the parameters name them.
    """

    @property
    def is_pretrained(self) -> bool:
        """
        Check whether the builders load published weights.

        Returns
        -------
        bool
            True if the parameters name published weights.
        """
        ...

    def without_pretrained_weights(self) -> Self:
        """
        Get the same architecture without published weights, for networks whose weights are restored from a saved
        checkpoint instead.

        Returns
        -------
        Self
            The parameters, building randomly initialized networks.
        """
        ...
