from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import TimesNetForecastNetwork
from .reconstruction_network import TimesNetReconstructionNetwork
from .summary_network import TimesNetSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class TimesNetParameters:
    """
    TimesNet (Wu et al., 2023): sequences folded into 2D grids at their dominant periods and convolved.

    The defaults follow the long-term forecasting scripts of the Time-Series-Library; no time features are embedded.

    Attributes
    ----------
    model_size : int
        Size of each token.
    feedforward_size : int
        Number of hidden channels of the inception convolutions.
    num_layers : int
        Number of TimesBlocks.
    top_k : int
        Largest number of dominant periods per block.
    num_kernels : int
        Number of parallel convolutions of each inception block.
    dropout : float
        Dropout probability of the embedding and of the summary head.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    model_size: int = 64
    feedforward_size: int = 64
    num_layers: int = 2
    top_k: int = 5
    num_kernels: int = 6
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("feedforward_size", self.feedforward_size),
            ("num_layers", self.num_layers),
            ("top_k", self.top_k),
            ("num_kernels", self.num_kernels),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

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
        return TimesNetForecastNetwork(self, shape)

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
        return TimesNetReconstructionNetwork(self, shape)

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
        return TimesNetSummaryNetwork(self, shape)
