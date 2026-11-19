from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....components.layers import Activation
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .sequence_encoder import TransformerSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class TransformerParameters:
    """
    Transformer encoder (Vaswani et al., 2017) with the generic heads of every task.

    Defaults follow the base model of the paper, as used by the Time-Series-Library.

    Attributes
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads; it divides model_size.
    num_layers : int
        Number of encoder layers.
    feedforward_size : int
        Hidden size of the position-wise feed-forward networks.
    dropout : float
        Dropout probability of the embedding, attention weights and sublayer outputs.
    activation : Activation
        Activation of the feed-forward networks.
    is_pre_norm : bool
        Whether layer normalization precedes each sublayer (with a final normalization) instead of following its
        residual connection as in the original Transformer.
    head : EncoderHead
        Output layers.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    model_size: int = 512
    num_heads: int = 8
    num_layers: int = 2
    feedforward_size: int = 2048
    dropout: float = 0.1
    activation: Activation = Activation.GELU
    is_pre_norm: bool = False
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("feedforward_size", self.feedforward_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads must divide model_size, got {self.num_heads} and {self.model_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_encoder(self, input_length: int, input_channels: int) -> TransformerSequenceEncoder:
        """
        Build the untrained Transformer encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        TransformerSequenceEncoder
            The encoder.
        """
        return TransformerSequenceEncoder(self, input_length, input_channels)

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
