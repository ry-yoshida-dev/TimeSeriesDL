from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import MOMENTEncoder
from .encoding import MOMENTEncoding

if TYPE_CHECKING:
    from .parameters import MOMENTParameters


class MOMENTForecastNetwork(ForecastNetwork):
    """
    MOMENT's forecasting head on each target channel as a separate series: the flattened patch states are mapped
    linearly to the horizon, followed by dropout, and the normalization is undone on the parameters in the scale of
    the values. The head is newly initialized, so the network must be trained (fine-tuned) before it forecasts;
    other channels are not read.

    Windows longer than ``input_length`` are cut to their last ``input_length`` values and shorter ones left-padded.

    Parameters
    ----------
    parameters : MOMENTParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: MOMENTParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.input_length: int = parameters.input_length
        self.encoder: MOMENTEncoder = MOMENTEncoder(parameters)
        self.head: nn.Linear = nn.Linear(
            parameters.num_patches * parameters.model_size, shape.horizon * shape.num_parameters
        )
        self.head_dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        is_level: Tensor = torch.zeros(shape.num_parameters, dtype=torch.bool)
        is_level[list(shape.level_parameters)] = True
        self.is_level: Tensor
        self.register_buffer("is_level", is_level, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch: int = past.shape[0]
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2).flatten(end_dim=1)
        padded, is_input = self.encoder.pad(series[:, -self.input_length :])
        encoding: MOMENTEncoding = self.encoder(padded, is_input, torch.ones_like(is_input), is_input)
        outputs: Tensor = self.head_dropout(self.head(encoding.states.flatten(start_dim=1)))
        parameters: Tensor = outputs.reshape(-1, self.horizon, self.num_parameters)
        parameters = torch.where(self.is_level, encoding.denormalize(parameters), parameters)
        return parameters.reshape(batch, -1, self.horizon, self.num_parameters).transpose(1, 2)
