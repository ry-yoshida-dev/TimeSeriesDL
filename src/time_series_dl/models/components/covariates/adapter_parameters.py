from dataclasses import dataclass, replace

from ...interfaces import CovariateForecastNetwork, CovariateForecastShape, ForecastingModel, ForecastShape
from .adapter_network import CovariateAdapterNetwork


@dataclass(frozen=True, kw_only=True)
class CovariateAdapterParameters:
    """
    Generic covariate support for forecasting models that do not read covariates natively.

    Attributes
    ----------
    hidden_size : int
        Width of the hidden layer of the per-step correction.
    dropout : float
        Dropout probability after the hidden layer of the correction.

    Raises
    ------
    ValueError
        If hidden_size is not positive or dropout is not in [0, 1).
    """

    hidden_size: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_network(self, model: ForecastingModel, shape: CovariateForecastShape) -> CovariateForecastNetwork:
        """
        Build an untrained adapter around a network of the model.

        Parameters
        ----------
        model : ForecastingModel
            The model whose forecast network is wrapped.
        shape : CovariateForecastShape
            The dimensions of the forecast and of the covariates.

        Returns
        -------
        CovariateForecastNetwork
            The adapter.
        """
        base_shape: ForecastShape = replace(
            shape.forecast, input_channels=shape.forecast.input_channels + shape.known_channels
        )
        return CovariateAdapterNetwork(model.build_forecast_network(base_shape), shape, self)
