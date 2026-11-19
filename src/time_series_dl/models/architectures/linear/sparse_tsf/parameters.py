from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import SparseTSFForecastNetwork


@dataclass(frozen=True, kw_only=True)
class SparseTSFParameters:
    """
    SparseTSF (Lin et al., 2024): cross-period sparse forecasting, one linear map from the past periods to the
    future periods shared by every phase of the period.

    The default period follows the official scripts for hourly data with a daily cycle.

    Attributes
    ----------
    period : int
        Length of the period in time steps.

    Raises
    ------
    ValueError
        If period is not positive.
    """

    period: int = 24

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")

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
        return SparseTSFForecastNetwork(self, shape)
