from typing import override

import torch
from deep_learning_loss import PointLoss
from torch import Tensor

from .....models import ForecastNetwork, ForecastShape
from .....models.interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from ....forecasting import InstanceNormalization, PointObjective
from ....forecasting.forecast_step import ForecastStep


class ForecastErrorScoringNetwork(AnomalyScoringNetwork):
    """
    Forecasts the last ``horizon`` steps of each window from the steps before them, trains on the forecast loss and
    scores each forecast step by its point loss averaged over channels; the past steps are not scored (NaN).

    Parameters
    ----------
    network : ForecastNetwork
        The point forecast network of the model, forecasting every channel.
    loss : PointLoss
        The loss of each forecast value.
    shape : ForecastShape
        The dimensions the network was built with; every input channel must be a target.
    is_instance_normalized : bool
        Whether each past window is normalized by its own mean and standard deviation, as in forecasting.

    Raises
    ------
    ValueError
        If the shape does not forecast every input channel in order.
    """

    def __init__(
        self, network: ForecastNetwork, loss: PointLoss, shape: ForecastShape, is_instance_normalized: bool
    ) -> None:
        if not shape.is_every_channel_target:
            raise ValueError(f"every input channel must be forecast in order, got targets {shape.target_channels}")
        super().__init__()
        self.network: ForecastNetwork = network
        self.loss: PointLoss = loss
        self.input_length: int = shape.input_length
        self._step: ForecastStep = ForecastStep(
            network=network,
            objective=PointObjective(loss),
            target_channels=shape.target_channels,
            normalization=InstanceNormalization() if is_instance_normalized else None,
        )

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        past: Tensor = values[:, : self.input_length]
        future: Tensor = values[:, self.input_length :]
        return self._step.compute_loss((past, future), is_training=self.training)

    @override
    def forward(self, values: Tensor) -> Tensor:
        past: Tensor = values[:, : self.input_length]
        future: Tensor = values[:, self.input_length :]
        forecast: Tensor = self._step.compute_parameters((past,))[..., 0]
        errors: Tensor = self.loss.compute_elementwise(forecast, future).mean(dim=2)
        unscored: Tensor = torch.full((values.shape[0], self.input_length), torch.nan, dtype=errors.dtype)
        return torch.cat([unscored.to(errors.device), errors], dim=1)
