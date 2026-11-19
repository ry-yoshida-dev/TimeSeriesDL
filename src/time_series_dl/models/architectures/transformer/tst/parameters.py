from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....components.layers import Activation, TemporalPooling
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from ..patch_tst import TokenNormalization
from .sequence_encoder import TSTSequenceEncoder


def _build_default_head() -> EncoderHead:
    return EncoderHead(pooling=TemporalPooling.FLATTEN, dropout=0.1)


@dataclass(frozen=True, kw_only=True)
class TSTParameters:
    """
    TST (Zerveas et al., KDD 2021): a Transformer encoder over projected time steps with a learnable position encoding
    and batch normalization, with the generic heads of every task.

    The default head flattens every step's features after dropout, as the official classification and regression
    output layer; reconstructions map each step's features to its values, as the official imputation output layer.
    Defaults follow the official code.

    Attributes
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads; must divide model_size.
    num_layers : int
        Number of Transformer encoder layers.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    dropout : float
        Dropout probability of the position encoding, the attention weights and the sublayer outputs.
    activation : Activation
        Activation of the feed-forward networks and of the encoder output.
    normalization : TokenNormalization
        Normalization of the tokens inside the encoder.
    head : EncoderHead
        Output layers of every task.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    model_size: int = 64
    num_heads: int = 8
    num_layers: int = 3
    feedforward_size: int = 256
    dropout: float = 0.1
    activation: Activation = Activation.GELU
    normalization: TokenNormalization = TokenNormalization.BATCH
    head: EncoderHead = field(default_factory=_build_default_head)

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
            raise ValueError(f"num_heads {self.num_heads} must divide model_size {self.model_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_encoder(self, input_length: int, input_channels: int) -> TSTSequenceEncoder:
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
        TSTSequenceEncoder
            The encoder.
        """
        return TSTSequenceEncoder(self, input_length, input_channels)

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
