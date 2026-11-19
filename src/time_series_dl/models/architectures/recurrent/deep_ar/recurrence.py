from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .parameters import DeepARParameters

type RecurrentState = Tensor | tuple[Tensor, Tensor]


class DeepARRecurrence(nn.Module):
    """
    Recurrent layers of DeepAR that encode the past window and then decode the horizon one step at a time.

    Past steps and decoded steps have their own linear input projections into the same recurrent layers. Each decoded
    step reads the previous step's target values together with its own features (such as covariates): the given
    future targets while training, and the predicted location parameter otherwise.

    Parameters
    ----------
    parameters : DeepARParameters
        The model configuration.
    past_size : int
        Number of inputs of each past step.
    output_channels : int
        Number of target channels.
    step_feature_size : int
        Number of features of each horizon step read next to the previous targets.
    num_parameters : int
        Number of distribution parameters per forecast value.
    location_index : int
        Index of the parameter fed back as the next step's previous target.
    """

    def __init__(
        self,
        parameters: DeepARParameters,
        past_size: int,
        output_channels: int,
        step_feature_size: int,
        num_parameters: int,
        location_index: int,
    ) -> None:
        super().__init__()
        self.output_channels: int = output_channels
        self.num_parameters: int = num_parameters
        self.location_index: int = location_index
        self.past_projection: nn.Linear = nn.Linear(past_size, parameters.hidden_size)
        self.target_projection: nn.Linear = nn.Linear(output_channels + step_feature_size, parameters.hidden_size)
        self.recurrent: nn.LSTM | nn.GRU | nn.RNN = parameters.cell.build(
            parameters.hidden_size, parameters.hidden_size, parameters.num_layers, parameters.dropout, False
        )
        self.head: nn.Linear = nn.Linear(parameters.hidden_size, output_channels * num_parameters)

    def forward(
        self, past: Tensor, last_targets: Tensor, step_features: Tensor, future_targets: Tensor | None
    ) -> Tensor:
        """
        Forecast the horizon.

        Parameters
        ----------
        past : Tensor
            Inputs of the past steps with shape (batch, input_length, past_size).
        last_targets : Tensor
            Target values of the last past step with shape (batch, 1, output_channels).
        step_features : Tensor
            Features of each horizon step with shape (batch, horizon, step_feature_size).
        future_targets : Tensor | None
            True future targets with shape (batch, horizon, output_channels) for teacher forcing; None to feed back
            the predicted locations.

        Returns
        -------
        Tensor
            Unconstrained distribution parameters with shape (batch, horizon, output_channels, num_parameters).

        Raises
        ------
        ValueError
            If the future targets do not have the shape of the horizon.
        """
        horizon: int = step_features.shape[1]
        expected: tuple[int, int, int] = (past.shape[0], horizon, self.output_channels)
        if future_targets is not None and tuple(future_targets.shape) != expected:
            raise ValueError(f"future_targets must have shape {expected}, got {tuple(future_targets.shape)}")
        state: RecurrentState = self.recurrent(self.past_projection(past))[1]
        if future_targets is not None:
            previous_targets: Tensor = torch.cat([last_targets, future_targets[:, :-1, :]], dim=1)
            return self._decode(torch.cat([previous_targets, step_features], dim=-1), state)[0]
        previous: Tensor = last_targets
        steps: list[Tensor] = []
        for position in range(horizon):
            step: Tensor
            step, state = self._decode(torch.cat([previous, step_features[:, position : position + 1]], dim=-1), state)
            steps.append(step)
            previous = step[..., self.location_index]
        return torch.cat(steps, dim=1)

    def _decode(self, inputs: Tensor, state: RecurrentState) -> tuple[Tensor, RecurrentState]:
        hidden: Tensor
        next_state: RecurrentState
        hidden, next_state = self.recurrent(self.target_projection(inputs), state)
        outputs: Tensor = self.head(hidden).reshape(
            hidden.shape[0], hidden.shape[1], self.output_channels, self.num_parameters
        )
        return outputs, next_state

    def __call__(
        self, past: Tensor, last_targets: Tensor, step_features: Tensor, future_targets: Tensor | None
    ) -> Tensor:
        return cast(Tensor, super().__call__(past, last_targets, step_features, future_targets))
