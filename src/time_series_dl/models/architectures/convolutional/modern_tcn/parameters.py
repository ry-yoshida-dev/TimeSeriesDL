from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import ModernTCNForecastNetwork
from .reconstruction_network import ModernTCNReconstructionNetwork
from .stage import ModernTCNStage
from .summary_network import ModernTCNSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class ModernTCNParameters:
    """
    ModernTCN (Luo and Wang, 2024): patches of each variable mixed by large-kernel depthwise convolutions along time
    and by pointwise feed-forward networks across features and across variables.

    The defaults follow the official long-term forecasting script for ETTh1: one stage of one block.

    Attributes
    ----------
    stages : tuple[ModernTCNStage, ...]
        The stages, from the patch embedding to the output; each later stage first downsamples its input.
    patch_size : int
        Number of time steps per patch.
    patch_stride : int
        Number of time steps between the starts of consecutive patches; at most patch_size.
    downsample_ratio : int
        Kernel length and stride of the downsampling between stages.
    feedforward_ratio : int
        Ratio of the hidden size of each feed-forward network to its input size.
    dropout : float
        Dropout probability inside the blocks.
    head_dropout : float
        Dropout probability before the output layer.

    Raises
    ------
    ValueError
        If stages is empty, a size is not positive, patch_stride exceeds patch_size, or a dropout is not in [0, 1).
    """

    stages: tuple[ModernTCNStage, ...] = (ModernTCNStage(),)
    patch_size: int = 8
    patch_stride: int = 4
    downsample_ratio: int = 2
    feedforward_ratio: int = 8
    dropout: float = 0.3
    head_dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("stages must not be empty.")
        for name, value in (
            ("patch_size", self.patch_size),
            ("patch_stride", self.patch_stride),
            ("downsample_ratio", self.downsample_ratio),
            ("feedforward_ratio", self.feedforward_ratio),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.patch_stride > self.patch_size:
            raise ValueError(f"patch_stride {self.patch_stride} exceeds patch_size {self.patch_size}")
        for name, probability in (("dropout", self.dropout), ("head_dropout", self.head_dropout)):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {probability}")

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
        return ModernTCNForecastNetwork(self, shape)

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
        return ModernTCNReconstructionNetwork(self, shape)

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
        return ModernTCNSummaryNetwork(self, shape)
