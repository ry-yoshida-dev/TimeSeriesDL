from dataclasses import dataclass, field

from ....components.heads import EncoderHead
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .absolute_position_encoding import AbsolutePositionEncoding
from .relative_position_encoding import RelativePositionEncoding
from .sequence_encoder import ConvTranSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class ConvTranParameters:
    """
    ConvTran (Foumani et al., 2024): convolutional embedding and one attention block with time absolute (tAPE) and
    efficient relative (eRPE) position encodings, with the generic heads of every task.

    The default head averages the features over time, as the paper's global average pooling before the output layer.
    Defaults follow the official code.

    Attributes
    ----------
    model_size : int
        Size of each token vector (the embedding size).
    num_heads : int
        Number of attention heads; must divide model_size.
    feedforward_size : int
        Hidden size of the feed-forward network.
    kernel_size : int
        Number of time steps of the temporal convolution of the embedding.
    dropout : float
        Dropout probability of the position encoding, the attention weights and the feed-forward network.
    absolute_position_encoding : AbsolutePositionEncoding
        Absolute position encoding of the attention inputs.
    relative_position_encoding : RelativePositionEncoding
        Relative position encoding inside the attention.
    head : EncoderHead
        Output layers of every task.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    model_size: int = 16
    num_heads: int = 8
    feedforward_size: int = 256
    kernel_size: int = 8
    dropout: float = 0.01
    absolute_position_encoding: AbsolutePositionEncoding = AbsolutePositionEncoding.TAPE
    relative_position_encoding: RelativePositionEncoding = RelativePositionEncoding.ERPE
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("feedforward_size", self.feedforward_size),
            ("kernel_size", self.kernel_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must divide model_size {self.model_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_encoder(self, input_length: int, input_channels: int) -> ConvTranSequenceEncoder:
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
        ConvTranSequenceEncoder
            The encoder.
        """
        return ConvTranSequenceEncoder(self, input_length, input_channels)

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
