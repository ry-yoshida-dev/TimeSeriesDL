"""mTAN parameters that pretend to start from published weights, for the irregular tasks' restoring tests."""

from dataclasses import dataclass, replace
from typing import Self, override

from time_series_dl.models.architectures.continuous_time import MTANParameters
from time_series_dl.models.components.pretrained import HubClient, PretrainedWeights
from time_series_dl.models.interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from time_series_dl.models.interfaces.irregular_summary_network import IrregularSummaryNetwork
from time_series_dl.models.interfaces.shapes.irregular_sequence import IrregularSequenceShape
from time_series_dl.models.interfaces.shapes.irregular_summary import IrregularSummaryShape


@dataclass(frozen=True, kw_only=True)
class PublishedMTANParameters(MTANParameters):
    """
    mTAN whose builders request ``weights`` from the Hugging Face Hub, as models with published weights do, and then
    initialize randomly; no irregular model has published weights yet.

    Attributes
    ----------
    weights : PretrainedWeights | None
        The weights requested by every builder; None for none.
    """

    weights: PretrainedWeights | None = None

    @property
    def is_pretrained(self) -> bool:
        """
        Check whether the builders request published weights.

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
    def build_irregular_summary_network(self, shape: IrregularSummaryShape) -> IrregularSummaryNetwork:
        self._request_weights()
        return super().build_irregular_summary_network(shape)

    @override
    def build_irregular_interpolation_network(self, shape: IrregularSequenceShape) -> IrregularInterpolationNetwork:
        self._request_weights()
        return super().build_irregular_interpolation_network(shape)

    def _request_weights(self) -> None:
        if self.weights is not None:
            _ = HubClient.download(self.weights)
