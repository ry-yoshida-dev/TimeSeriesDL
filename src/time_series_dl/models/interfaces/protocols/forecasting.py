"""Structural type for models that build forecast networks."""

from typing import Protocol, runtime_checkable

from ..forecast_network import ForecastNetwork
from ..shapes import ForecastShape


@runtime_checkable
class ForecastingModel(Protocol):
    """
    Model parameters able to build a forecast network.
    """

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
        ...
