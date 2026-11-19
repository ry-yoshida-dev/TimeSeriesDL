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
from .sequence_encoder import TCNSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class TCNParameters:
    """
    Temporal convolutional network (Bai et al., 2018) with the generic heads of every task.

    The receptive field is ``1 + 2 * (kernel_size - 1) * (2 ** len(channel_sizes) - 1)`` steps.

    Attributes
    ----------
    channel_sizes : tuple[int, ...]
        Output channels of each residual block; block i has dilation 2**i.
    kernel_size : int
        Length of the convolution kernels.
    dropout : float
        Dropout probability after each convolution.
    head : EncoderHead
        Output layers; summaries use the last step by default.

    Raises
    ------
    ValueError
        If there is no block, a size is not positive, or dropout is not in [0, 1).
    """

    channel_sizes: tuple[int, ...] = (64, 64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    head: EncoderHead = field(default_factory=lambda: EncoderHead(pooling=TemporalPooling.LAST))

    def __post_init__(self) -> None:
        if not self.channel_sizes or any(size <= 0 for size in self.channel_sizes):
            raise ValueError(f"channel_sizes must be non-empty and positive, got {self.channel_sizes}")
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_encoder(self, input_length: int, input_channels: int) -> TCNSequenceEncoder:
        """
        Build the untrained convolutional encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        TCNSequenceEncoder
            The encoder.
        """
        return TCNSequenceEncoder(self, input_length, input_channels)

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
