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
from .minimal_recurrent_cell import MinimalRecurrentCell
from .sequence_encoder import MinRNNSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class MinRNNParameters:
    """
    minGRU / minLSTM (Feng et al., 2024): residual blocks of minimal recurrent layers trained with a parallel scan,
    with the generic heads of every task.

    The paper gives no time series configuration; the defaults are a small pre-normalized residual stack with a
    feedforward expansion of 4, as in the language modeling setup the paper follows.

    Attributes
    ----------
    cell : MinimalRecurrentCell
        minGRU or minLSTM.
    model_size : int
        Size of the embedding and of the features between blocks.
    hidden_size : int
        Size of the hidden state of each recurrent layer.
    num_layers : int
        Number of residual blocks.
    feedforward_factor : int
        Ratio of the hidden size of each feedforward network to model_size.
    dropout : float
        Dropout probability of the output of each recurrent layer and feedforward network.
    head : EncoderHead
        Output layers; summaries use the last step by default, since the blocks are causal.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    cell: MinimalRecurrentCell = MinimalRecurrentCell.GRU
    model_size: int = 128
    hidden_size: int = 128
    num_layers: int = 3
    feedforward_factor: int = 4
    dropout: float = 0.1
    head: EncoderHead = field(default_factory=lambda: EncoderHead(pooling=TemporalPooling.LAST))

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("feedforward_factor", self.feedforward_factor),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_encoder(self, input_length: int, input_channels: int) -> MinRNNSequenceEncoder:
        """
        Build the untrained encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        MinRNNSequenceEncoder
            The encoder.
        """
        return MinRNNSequenceEncoder(self, input_length, input_channels)

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
        Build an untrained reconstruction network.

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
