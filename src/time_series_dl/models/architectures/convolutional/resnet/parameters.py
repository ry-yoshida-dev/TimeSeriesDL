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
from .sequence_encoder import ResNetSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class ResNetParameters:
    """
    Residual network (Wang et al., 2017) with the generic heads of every task.

    The defaults are the three residual blocks of the paper; with the default head, summaries use global average
    pooling as in the paper.

    Attributes
    ----------
    channel_sizes : tuple[int, ...]
        Number of filters of each residual block.
    kernel_sizes : tuple[int, ...]
        Kernel length of the successive convolutions inside every block; even lengths are allowed.
    head : EncoderHead
        Output layers; summaries average over time by default.

    Raises
    ------
    ValueError
        If there is no block or no convolution per block, or a size is not positive.
    """

    channel_sizes: tuple[int, ...] = (64, 128, 128)
    kernel_sizes: tuple[int, ...] = (8, 5, 3)
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        if not self.channel_sizes or any(size <= 0 for size in self.channel_sizes):
            raise ValueError(f"channel_sizes must be non-empty and positive, got {self.channel_sizes}")
        if not self.kernel_sizes or any(size <= 0 for size in self.kernel_sizes):
            raise ValueError(f"kernel_sizes must be non-empty and positive, got {self.kernel_sizes}")

    def build_encoder(self, input_length: int, input_channels: int) -> ResNetSequenceEncoder:
        """
        Build the untrained residual encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        ResNetSequenceEncoder
            The encoder.
        """
        return ResNetSequenceEncoder(self, input_length, input_channels)

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
