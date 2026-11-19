"""Structural type for pretrained models that forecast without training."""

from typing import Protocol, runtime_checkable

from .forecasting import ForecastingModel
from .published_weights import PublishedWeightsModel


@runtime_checkable
class ZeroShotForecastingModel(ForecastingModel, PublishedWeightsModel, Protocol):
    """
    Model parameters whose forecast networks, built with published weights, forecast without training.

    Such a network outputs a point forecast when built for one parameter per value, and the quantiles of
    ``zero_shot_quantile_levels`` when built for that many parameters.
    """

    @property
    def zero_shot_quantile_levels(self) -> tuple[float, ...]:
        """
        Get the levels of the quantiles the forecast network outputs.

        Returns
        -------
        tuple[float, ...]
            The levels in increasing order.
        """
        ...
