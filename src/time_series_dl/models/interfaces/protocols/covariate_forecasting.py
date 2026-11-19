"""Structural type for models that read covariates natively."""

from typing import Protocol, runtime_checkable

from ..covariate_forecast_network import CovariateForecastNetwork
from ..shapes import CovariateForecastShape
from .forecasting import ForecastingModel


@runtime_checkable
class CovariateForecastingModel(ForecastingModel, Protocol):
    """
    Model parameters able to build a forecast network that reads known and static covariates in the way of its
    paper. Forecasting models without this builder still accept covariates through the task's generic adapter.
    """

    def build_covariate_forecast_network(self, shape: CovariateForecastShape) -> CovariateForecastNetwork:
        """
        Build an untrained covariate forecast network.

        Parameters
        ----------
        shape : CovariateForecastShape
            The input, covariate and output dimensions.

        Returns
        -------
        CovariateForecastNetwork
            The network.
        """
        ...
