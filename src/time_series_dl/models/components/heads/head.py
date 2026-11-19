from dataclasses import dataclass

from ...interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceEncoder,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from ..layers import TemporalPooling
from .forecast_network import EncodedForecastNetwork
from .reconstruction_network import EncodedReconstructionNetwork
from .summary_network import EncodedSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class EncoderHead:
    """
    Output layers that turn any sequence encoder into a network of every task.

    Attributes
    ----------
    pooling : TemporalPooling
        Reduction of the encoded sequence for summary networks.
    dropout : float
        Dropout probability before the output layer.

    Raises
    ------
    ValueError
        If dropout is not in [0, 1).
    """

    pooling: TemporalPooling = TemporalPooling.MEAN
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_forecast_network(self, encoder: SequenceEncoder, shape: ForecastShape) -> ForecastNetwork:
        """
        Attach a linear forecast head.

        Parameters
        ----------
        encoder : SequenceEncoder
            The encoder of the past values.
        shape : ForecastShape
            The dimensions of the forecast.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return EncodedForecastNetwork(encoder, shape, self.dropout)

    def build_reconstruction_network(self, encoder: SequenceEncoder, shape: SequenceShape) -> ReconstructionNetwork:
        """
        Attach a per-step reconstruction head.

        Parameters
        ----------
        encoder : SequenceEncoder
            The encoder; it must keep the length of its input.
        shape : SequenceShape
            The dimensions of the sequences.

        Returns
        -------
        ReconstructionNetwork
            The network.
        """
        return EncodedReconstructionNetwork(encoder, shape, self.dropout)

    def build_summary_network(self, encoder: SequenceEncoder, shape: SummaryShape) -> SummaryNetwork:
        """
        Attach a pooling and output head.

        Parameters
        ----------
        encoder : SequenceEncoder
            The encoder of the sequence.
        shape : SummaryShape
            The input and output dimensions.

        Returns
        -------
        SummaryNetwork
            The network.
        """
        return EncodedSummaryNetwork(encoder, shape, self.pooling, self.dropout)
