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
from .sequence_encoder import InceptionTimeSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class InceptionTimeParameters:
    """
    One InceptionTime network (Ismail Fawaz et al., 2020) with the generic heads of every task.

    The paper averages an ensemble of five such networks trained from different initializations; this is a single
    network. The defaults are those of the paper.

    Attributes
    ----------
    num_filters : int
        Number of filters of each branch of an Inception module.
    bottleneck_size : int
        Number of channels of the 1x1 bottleneck in front of the convolution branches, used when the module input has
        more than one channel.
    max_kernel_size : int
        Kernel length of the longest branch; branch i uses ``max_kernel_size // 2**i``, even lengths included.
    num_kernel_sizes : int
        Number of convolution branches, beside the max-pooling branch.
    depth : int
        Number of Inception modules.
    is_residual : bool
        Whether a shortcut is added around every ``residual_interval`` modules.
    residual_interval : int
        Number of modules spanned by each shortcut.
    head : EncoderHead
        Output layers; summaries average over time by default, the global average pooling of the paper.

    Raises
    ------
    ValueError
        If a size is not positive or the shortest kernel would be shorter than one step.
    """

    num_filters: int = 32
    bottleneck_size: int = 32
    max_kernel_size: int = 40
    num_kernel_sizes: int = 3
    depth: int = 6
    is_residual: bool = True
    residual_interval: int = 3
    head: EncoderHead = field(default_factory=EncoderHead)

    def __post_init__(self) -> None:
        for name, value in (
            ("num_filters", self.num_filters),
            ("bottleneck_size", self.bottleneck_size),
            ("max_kernel_size", self.max_kernel_size),
            ("num_kernel_sizes", self.num_kernel_sizes),
            ("depth", self.depth),
            ("residual_interval", self.residual_interval),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.max_kernel_size < 2 ** (self.num_kernel_sizes - 1):
            raise ValueError(
                f"max_kernel_size must be at least {2 ** (self.num_kernel_sizes - 1)} for "
                + f"{self.num_kernel_sizes} kernel sizes, got {self.max_kernel_size}"
            )

    @property
    def kernel_sizes(self) -> tuple[int, ...]:
        """
        Get the kernel length of each convolution branch.

        Returns
        -------
        tuple[int, ...]
            The lengths, halving from ``max_kernel_size``.
        """
        return tuple(self.max_kernel_size // 2**branch for branch in range(self.num_kernel_sizes))

    @property
    def module_size(self) -> int:
        """
        Get the number of output channels of each Inception module.

        Returns
        -------
        int
            The filters of every convolution branch and of the max-pooling branch.
        """
        return self.num_filters * (self.num_kernel_sizes + 1)

    def build_encoder(self, input_length: int, input_channels: int) -> InceptionTimeSequenceEncoder:
        """
        Build the untrained Inception encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        InceptionTimeSequenceEncoder
            The encoder.
        """
        return InceptionTimeSequenceEncoder(self, input_length, input_channels)

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
