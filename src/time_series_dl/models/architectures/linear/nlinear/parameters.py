from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import NLinearForecastNetwork


@dataclass(frozen=True, kw_only=True)
class NLinearParameters:
    """
    NLinear (Zeng et al., 2023): one linear map along time of the window minus its last value, which is added back.

    Attributes
    ----------
    is_individual : bool
        Whether each channel has its own linear map.
    """

    is_individual: bool = False

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
        return NLinearForecastNetwork(self, shape)
