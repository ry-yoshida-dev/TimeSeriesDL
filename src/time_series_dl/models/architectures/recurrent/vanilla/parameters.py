from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....components.layers import TemporalPooling
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .recurrent_cell import RecurrentCell
from .sequence_encoder import RNNSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class RNNParameters:
    """
    Recurrent network (LSTM, GRU or Elman) with the generic heads of every task.

    Attributes
    ----------
    cell : RecurrentCell
        Recurrent cell.
    hidden_size : int
        Size of the hidden state per direction.
    num_layers : int
        Number of stacked recurrent layers.
    dropout : float
        Dropout probability between recurrent layers.
    is_bidirectional : bool
        Whether each layer also reads the sequence backward; not causal, so the last step of a summary sees only one
        step of the backward direction.
    head : EncoderHead
        Output layers; summaries use the last hidden state by default.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    cell: RecurrentCell = RecurrentCell.LSTM
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    is_bidirectional: bool = False
    head: EncoderHead = field(default_factory=lambda: EncoderHead(pooling=TemporalPooling.LAST))

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_encoder(self, input_length: int, input_channels: int) -> RNNSequenceEncoder:
        """
        Build the untrained recurrent encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        RNNSequenceEncoder
            The encoder.
        """
        return RNNSequenceEncoder(self, input_length, input_channels)

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return self.head.build_forecast_network(self.build_encoder(shape.input_length, shape.input_channels), shape)

    def build_reconstruction_network(self, shape: SequenceShape) -> ReconstructionNetwork:
        """
        Build an untrained reconstruction network (an LSTM autoencoder with the default cell).

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences.

        Returns
        -------
        ReconstructionNetwork
            The network.
        """
        return self.head.build_reconstruction_network(self.build_encoder(shape.length, shape.channels), shape)

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
        return self.head.build_summary_network(self.build_encoder(shape.input_length, shape.input_channels), shape)
