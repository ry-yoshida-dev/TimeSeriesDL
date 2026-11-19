"""TS2Vec parameters that pretend to start from published weights, for the representation learning restoring test."""

from dataclasses import dataclass, replace
from typing import Self, override

from time_series_dl.models.architectures.convolutional.ts2vec import TS2VecParameters
from time_series_dl.models.architectures.convolutional.ts2vec.pretraining_network import TS2VecPretrainingNetwork
from time_series_dl.models.components.pretrained import HubClient, PretrainedWeights
from time_series_dl.models.interfaces import SequenceShape


@dataclass(frozen=True, kw_only=True)
class PublishedTS2VecParameters(TS2VecParameters):
    """
    TS2Vec whose pretraining builder requests ``weights`` from the Hugging Face Hub, as models with published weights
    do, and then initializes randomly; no pretraining method has published weights yet.

    Attributes
    ----------
    weights : PretrainedWeights | None
        The weights requested by the builder; None for none.
    """

    weights: PretrainedWeights | None = None

    @property
    def is_pretrained(self) -> bool:
        """
        Check whether the builder requests published weights.

        Returns
        -------
        bool
            True if ``weights`` is set.
        """
        return self.weights is not None

    def without_pretrained_weights(self) -> Self:
        """
        Get the same architecture without published weights.

        Returns
        -------
        Self
            The parameters with ``weights`` set to None.
        """
        return replace(self, weights=None)

    @override
    def build_pretraining_network(self, shape: SequenceShape) -> TS2VecPretrainingNetwork:
        if self.weights is not None:
            _ = HubClient.download(self.weights)
        return super().build_pretraining_network(shape)
