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
from .sequence_encoder import LITESequenceEncoder

_MIN_KERNEL_SIZE: int = 4


@dataclass(frozen=True, kw_only=True)
class LITEParameters:
    """
    One LITE network (Ismail-Fawaz et al., 2023): a light InceptionTime with custom filters, multiplexing
    convolutions and dilated depthwise separable convolutions, with the generic heads of every task.

    LITETime averages an ensemble of five LITE networks trained from different initializations; this is a single
    network. The defaults are those of the paper.

    Attributes
    ----------
    num_filters : int
        Number of filters of each multiplexing branch and of each separable convolution.
    kernel_size : int
        Kernel length of the longest multiplexing branch; the others halve it, and the separable convolutions use a
        half and a quarter of it.
    is_custom_filtered : bool
        Whether the fixed custom filters are concatenated to the first module.
    is_dilated : bool
        Whether the separable convolutions have dilations 2 and 4.
    is_multiplexed : bool
        Whether the first module has three branches of halving lengths instead of one.
    head : EncoderHead
        Output layers; summaries average over time by default, the global average pooling of the paper.

    Raises
    ------
    ValueError
        If num_filters is not positive or kernel_size is below 4.
    """

    num_filters: int = 32
    kernel_size: int = 41
    is_custom_filtered: bool = True
    is_dilated: bool = True
    is_multiplexed: bool = True
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        if self.num_filters <= 0:
            raise ValueError(f"num_filters must be positive, got {self.num_filters}")
        if self.kernel_size < _MIN_KERNEL_SIZE:
            raise ValueError(f"kernel_size must be at least {_MIN_KERNEL_SIZE}, got {self.kernel_size}")

    def build_encoder(self, input_length: int, input_channels: int) -> LITESequenceEncoder:
        """
        Build the untrained LITE encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        LITESequenceEncoder
            The encoder.
        """
        return LITESequenceEncoder(self, input_length, input_channels)

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
