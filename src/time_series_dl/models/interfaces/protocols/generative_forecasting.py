"""Structural type for models that forecast by drawing sample paths."""

from typing import Protocol, runtime_checkable

from ..generative_forecast_network import GenerativeForecastNetwork
from ..shapes import ForecastShape
from .forecasting import ForecastingModel


@runtime_checkable
class GenerativeForecastingModel(ForecastingModel, Protocol):
    """
    Model parameters able to build a generative forecast network, which draws sample paths and is trained with its
    own loss. Its ``build_forecast_network`` builds the same network, whose ``num_parameters`` are the samples.
    """

    def build_generative_forecast_network(self, shape: ForecastShape) -> GenerativeForecastNetwork:
        """
        Build an untrained generative forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths of ``forward``.

        Returns
        -------
        GenerativeForecastNetwork
            The network.
        """
        ...
