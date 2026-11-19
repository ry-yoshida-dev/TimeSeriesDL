from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import MICNForecastNetwork
from .reconstruction_network import MICNReconstructionNetwork
from .summary_network import MICNSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class MICNParameters:
    """
    MICN (Wang et al., 2023): multi-scale decomposition into trend and seasonal parts, a linear trend forecast, and
    multi-scale isometric convolutions that extract local features and global correlations of the seasonal part.

    The layers follow the Time-Series-Library; the defaults are those of the paper.

    Attributes
    ----------
    model_size : int
        Size of each token.
    num_layers : int
        Number of MIC layers.
    kernel_sizes : tuple[int, ...]
        Length and stride of the downsampling convolution of each scale; even lengths get decomposition windows one
        step longer, since those must be odd.
    dropout : float
        Dropout probability of the embedding, of every convolution and of the summary head.

    Raises
    ------
    ValueError
        If a size is not positive, kernel_sizes is empty, or dropout is not in [0, 1).
    """

    model_size: int = 512
    num_layers: int = 1
    kernel_sizes: tuple[int, ...] = (12, 16)
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.model_size <= 0:
            raise ValueError(f"model_size must be positive, got {self.model_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not self.kernel_sizes:
            raise ValueError("kernel_sizes must not be empty.")
        if any(size <= 0 for size in self.kernel_sizes):
            raise ValueError(f"kernel_sizes must be positive, got {self.kernel_sizes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def decomposition_sizes(self) -> tuple[int, ...]:
        """
        Get the moving-average window of each scale.

        Returns
        -------
        tuple[int, ...]
            Each kernel size, plus one when it is even.
        """
        return tuple(size + 1 if size % 2 == 0 else size for size in self.kernel_sizes)

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
        return MICNForecastNetwork(self, shape)

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
        return MICNReconstructionNetwork(self, shape)

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
        return MICNSummaryNetwork(self, shape)
