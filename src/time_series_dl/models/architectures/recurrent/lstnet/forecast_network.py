from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .sequence_encoder import LSTNetSequenceEncoder

if TYPE_CHECKING:
    from .parameters import LSTNetParameters


class LSTNetForecastNetwork(ForecastNetwork):
    """
    LSTNet with direct multi-step outputs: a linear map from the encoded features to the whole horizon, plus the
    highway autoregressive component.

    The autoregressive component maps the last ``highway_window`` values of each target channel linearly to that
    channel's horizon, with weights shared by the channels, and is added to the neural forecast so that the output
    follows the scale of the input.

    Parameters
    ----------
    parameters : LSTNetParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If the kernel, the skip period or the highway window does not fit in the input window.
    """

    def __init__(self, parameters: LSTNetParameters, shape: ForecastShape) -> None:
        if parameters.highway_window > shape.input_length:
            raise ValueError(
                f"highway_window {parameters.highway_window} must not exceed input_length {shape.input_length}"
            )
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.highway_window: int = parameters.highway_window
        self.encoder: LSTNetSequenceEncoder = LSTNetSequenceEncoder(
            parameters, shape.input_length, shape.input_channels
        )
        self.projection: nn.Linear = nn.Linear(
            self.encoder.output_size, shape.horizon * shape.output_channels * shape.num_parameters
        )
        self.highway: nn.Linear | None = (
            nn.Linear(parameters.highway_window, shape.horizon * shape.num_parameters)
            if parameters.highway_window > 0
            else None
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        features: Tensor = self.encoder(past).squeeze(1)
        outputs: Tensor = self.projection(features).reshape(-1, self.horizon, self.output_channels, self.num_parameters)
        if self.highway is not None:
            recent: Tensor = past[:, -self.highway_window :].index_select(2, self.target_channels).transpose(1, 2)
            autoregressive: Tensor = self.highway(recent).reshape(*recent.shape[:-1], self.horizon, self.num_parameters)
            outputs = outputs + autoregressive.transpose(1, 2)
        return outputs
