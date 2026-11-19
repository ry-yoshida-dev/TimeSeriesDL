from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ...interfaces import CovariateForecastNetwork, CovariateForecastShape, CovariateTensors, ForecastNetwork
from ..layers import Activation, MultilayerPerceptron
from .static_embedding import StaticCovariateEmbedding

if TYPE_CHECKING:
    from .adapter_parameters import CovariateAdapterParameters


class CovariateAdapterNetwork(CovariateForecastNetwork):
    """
    Gives any forecast network covariates: the past known covariates become extra input channels of the network, and
    a per-step correction of its outputs reads the future known covariates and the static features.

    The wrapped network reads the past values followed by the known covariates of the past window, so it can relate
    the history of the target to that of the covariates. For each horizon step, a one-hidden-layer MLP reads the
    network's distribution parameters of that step, the known covariates of that step and the static features, and
    its output is added to the parameters. Since the MLP sees the network's own forecast, a covariate can shift it or
    rescale it, but its effect on one step only comes from that step's covariates and from the past window.

    Parameters
    ----------
    base : ForecastNetwork
        Network built for the input channels followed by the known covariate channels.
    shape : CovariateForecastShape
        The dimensions of the forecast and of the covariates.
    parameters : CovariateAdapterParameters
        Size and dropout of the correction.
    """

    def __init__(
        self, base: ForecastNetwork, shape: CovariateForecastShape, parameters: CovariateAdapterParameters
    ) -> None:
        super().__init__()
        forecast = shape.forecast
        self.base: ForecastNetwork = base
        self.input_length: int = forecast.input_length
        self.horizon: int = forecast.horizon
        self.static_embedding: StaticCovariateEmbedding = StaticCovariateEmbedding(
            shape.static_channels, shape.static_cardinalities
        )
        outputs: int = forecast.output_channels * forecast.num_parameters
        self.correction: MultilayerPerceptron = MultilayerPerceptron(
            outputs + shape.known_channels + self.static_embedding.output_size,
            (parameters.hidden_size,),
            outputs,
            Activation.RELU,
            parameters.dropout,
        )

    @override
    def forward(self, past: Tensor, covariates: CovariateTensors, future_targets: Tensor | None = None) -> Tensor:
        known: Tensor = covariates.known
        forecast: Tensor = self.base(torch.cat([past, known[:, : self.input_length]], dim=2), future_targets)
        static: Tensor = self.static_embedding(covariates.static, covariates.static_categories)
        features: Tensor = torch.cat(
            [
                forecast.flatten(start_dim=2),
                known[:, self.input_length :],
                static.unsqueeze(1).expand(-1, self.horizon, -1),
            ],
            dim=-1,
        )
        return forecast + self.correction(features).reshape(forecast.shape)
