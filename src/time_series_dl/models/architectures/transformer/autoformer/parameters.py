from dataclasses import dataclass, field

from ....components.attention.attention_mechanism import AttentionMechanism
from ....components.attention.attention_mechanisms.auto_correlation import AutoCorrelationAttentionMechanism
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
from ..decomposition import DecompositionTransformerForecastNetwork, DecompositionTransformerSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class AutoformerParameters:
    """
    Autoformer (Wu et al., NeurIPS 2021): auto-correlation in place of attention, inside a decomposition Transformer.

    Forecasts use the encoder-decoder; reconstructions and summaries use the encoder and the generic heads. Defaults
    follow the official code and its experiment scripts.

    Attributes
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of auto-correlation heads; it divides model_size.
    num_encoder_layers : int
        Number of encoder layers.
    num_decoder_layers : int
        Number of decoder layers.
    feedforward_size : int
        Hidden size of the position-wise feed-forward networks.
    moving_average_size : int
        Length of the moving average of the series decompositions; odd.
    factor : int
        Factor c of the ``c * ln L`` aggregated delays.
    dropout : float
        Dropout probability of the embeddings and sublayer outputs.
    activation : Activation
        Activation of the feed-forward networks.
    label_length : int
        Number of last past steps that start the decoder input; at most the input length.
    head : EncoderHead
        Output layers of reconstructions and summaries.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, moving_average_size is even, or dropout is
        not in [0, 1).
    """

    model_size: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 1
    feedforward_size: int = 2048
    moving_average_size: int = 25
    factor: int = 3
    dropout: float = 0.05
    activation: Activation = Activation.GELU
    label_length: int = 48
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_encoder_layers", self.num_encoder_layers),
            ("num_decoder_layers", self.num_decoder_layers),
            ("feedforward_size", self.feedforward_size),
            ("factor", self.factor),
            ("label_length", self.label_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads must divide model_size, got {self.num_heads} and {self.model_size}")
        if self.moving_average_size <= 0 or self.moving_average_size % 2 == 0:
            raise ValueError(f"moving_average_size must be a positive odd number, got {self.moving_average_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_self_attention(self, length: int) -> AttentionMechanism:
        """
        Build the auto-correlation of a sequence.

        Parameters
        ----------
        length : int
            Number of time steps of the sequence.

        Returns
        -------
        AttentionMechanism
            The mechanism.
        """
        return AutoCorrelationAttentionMechanism(self.factor)

    def build_cross_attention(self, query_length: int, key_length: int) -> AttentionMechanism:
        """
        Build the auto-correlation from the decoder to the encoder output.

        Parameters
        ----------
        query_length : int
            Number of time steps of the decoder.
        key_length : int
            Number of time steps of the encoder output.

        Returns
        -------
        AttentionMechanism
            The mechanism.
        """
        return AutoCorrelationAttentionMechanism(self.factor)

    def build_encoder(self, input_length: int, input_channels: int) -> DecompositionTransformerSequenceEncoder:
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
        DecompositionTransformerSequenceEncoder
            The encoder.
        """
        return DecompositionTransformerSequenceEncoder(self, input_length, input_channels)

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained encoder-decoder forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network.

        Raises
        ------
        ValueError
            If label_length exceeds the input length.
        """
        return DecompositionTransformerForecastNetwork(self, shape)

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
