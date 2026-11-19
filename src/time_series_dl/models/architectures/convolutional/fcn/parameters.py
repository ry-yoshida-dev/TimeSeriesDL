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
from .sequence_encoder import FCNSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class FCNParameters:
    """
    Fully convolutional network (Wang et al., 2017) with the generic heads of every task.

    The defaults are the three blocks of the paper; with the default head, summaries use global average pooling as in
    the paper.

    Attributes
    ----------
    channel_sizes : tuple[int, ...]
        Number of filters of each convolution block.
    kernel_sizes : tuple[int, ...]
        Kernel length of each convolution block; even lengths are allowed.
    head : EncoderHead
        Output layers; summaries average over time by default.

    Raises
    ------
    ValueError
        If there is no block, a size is not positive, or the two tuples differ in length.
    """

    channel_sizes: tuple[int, ...] = (128, 256, 128)
    kernel_sizes: tuple[int, ...] = (8, 5, 3)
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        if not self.channel_sizes or any(size <= 0 for size in self.channel_sizes):
            raise ValueError(f"channel_sizes must be non-empty and positive, got {self.channel_sizes}")
        if any(size <= 0 for size in self.kernel_sizes):
            raise ValueError(f"kernel_sizes must be positive, got {self.kernel_sizes}")
        if len(self.kernel_sizes) != len(self.channel_sizes):
            raise ValueError(
                f"kernel_sizes must have one length per block ({len(self.channel_sizes)}), got {self.kernel_sizes}"
            )

    def build_encoder(self, input_length: int, input_channels: int) -> FCNSequenceEncoder:
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
        FCNSequenceEncoder
            The encoder.
        """
        return FCNSequenceEncoder(self, input_length, input_channels)

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
