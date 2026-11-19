from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import PatchTSTEncoder

if TYPE_CHECKING:
    from .parameters import PatchTSTParameters


class PatchTSTForecastNetwork(ForecastNetwork):
    """
    Patch encoder of each target channel followed by a linear map of its flattened features to the horizon.

    Since the model is channel-independent, only the target channels are encoded; the other input channels cannot
    affect the forecast.

    Parameters
    ----------
    parameters : PatchTSTParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: PatchTSTParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.encoder: PatchTSTEncoder = PatchTSTEncoder(parameters, shape.input_length)
        self.dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.projection: nn.Linear = nn.Linear(self.encoder.feature_size, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        features: Tensor = self.encoder(past.index_select(2, self.target_channels))
        outputs: Tensor = self.projection(self.dropout(features.flatten(start_dim=2)))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
