from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import DLinearForecastNetwork
from .reconstruction_network import DLinearReconstructionNetwork
from .summary_network import DLinearSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class DLinearParameters:
    """
    DLinear (Zeng et al., 2023): one linear map along time for the trend and one for the seasonal remainder.

    Attributes
    ----------
    kernel_size : int
        Length of the moving average that extracts the trend; odd.
    is_individual : bool
        Whether each channel has its own linear maps.

    Raises
    ------
    ValueError
        If kernel_size is not a positive odd number.
    """

    kernel_size: int = 25
    is_individual: bool = False

    def __post_init__(self) -> None:
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd number, got {self.kernel_size}")

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
        return DLinearForecastNetwork(self, shape)

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
        return DLinearReconstructionNetwork(self, shape)

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
        return DLinearSummaryNetwork(self, shape)
