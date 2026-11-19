from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape

if TYPE_CHECKING:
    from .parameters import DeepARParameters

type RecurrentState = Tensor | tuple[Tensor, Tensor]


class DeepARForecastNetwork(ForecastNetwork):
    """
    Recurrent layers that encode the past window and then decode the horizon one step at a time.

    Past steps and decoded steps have their own linear input projections into the same recurrent layers. The first
    decoded step reads the last past target values; each later step reads the previous step's target values, which
    are the given future targets while training and the predicted location parameter otherwise.

    Parameters
    ----------
    parameters : DeepARParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: DeepARParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.location_index: int = shape.location_index
        self.past_projection: nn.Linear = nn.Linear(shape.input_channels, parameters.hidden_size)
        self.target_projection: nn.Linear = nn.Linear(shape.output_channels, parameters.hidden_size)
        self.recurrent: nn.LSTM | nn.GRU | nn.RNN = parameters.cell.build(
            parameters.hidden_size, parameters.hidden_size, parameters.num_layers, parameters.dropout, False
        )
        self.head: nn.Linear = nn.Linear(parameters.hidden_size, shape.output_channels * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch_size: int = past.shape[0]
        expected: tuple[int, int, int] = (batch_size, self.horizon, self.output_channels)
        if future_targets is not None and tuple(future_targets.shape) != expected:
            raise ValueError(f"future_targets must have shape {expected}, got {tuple(future_targets.shape)}")
        state: RecurrentState = self.recurrent(self.past_projection(past))[1]
        previous: Tensor = past.index_select(2, self.target_channels)[:, -1:, :]
        if future_targets is not None:
            inputs: Tensor = torch.cat([previous, future_targets[:, :-1, :]], dim=1)
            return self._decode(inputs, state)[0]
        steps: list[Tensor] = []
        for _ in range(self.horizon):
            step: Tensor
            step, state = self._decode(previous, state)
            steps.append(step)
            previous = step[..., self.location_index]
        return torch.cat(steps, dim=1)

    def _decode(self, previous_targets: Tensor, state: RecurrentState) -> tuple[Tensor, RecurrentState]:
        hidden: Tensor
        next_state: RecurrentState
        hidden, next_state = self.recurrent(self.target_projection(previous_targets), state)
        outputs: Tensor = self.head(hidden).reshape(
            hidden.shape[0], hidden.shape[1], self.output_channels, self.num_parameters
        )
        return outputs, next_state
