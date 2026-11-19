from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.diffusion import DiffusionProcess, NoisedSample
from ....interfaces import ForecastShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from .denoiser import TimeGradDenoiser

if TYPE_CHECKING:
    from .parameters import TimeGradParameters

type RecurrentState = Tensor | tuple[Tensor, Tensor]


class TimeGradForecastNetwork(GenerativeForecastNetwork):
    """
    TimeGrad: recurrent layers summarize the series up to each step, and a diffusion model conditioned on that
    summary draws the next step's target values.

    The past steps read every input channel; later steps read the previous step's target values. While training, the
    targets of every step after the first (in the past window and the horizon) are diffused to a random step and the
    denoiser predicts the noise from the hidden state of the step before. Sampling repeats each window's encoded state
    for every sample path and, at each horizon step, runs the reverse diffusion from the hidden state and feeds the
    drawn values back.

    Parameters
    ----------
    parameters : TimeGradParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast; ``num_parameters`` is the number of sample paths of ``forward``.
    """

    def __init__(self, parameters: TimeGradParameters, shape: ForecastShape) -> None:
        super().__init__(shape.num_parameters)
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.past_projection: nn.Linear = nn.Linear(shape.input_channels, parameters.hidden_size)
        # A one-step horizon never reads previous targets.
        self.target_projection: nn.Linear | None = (
            nn.Linear(shape.output_channels, parameters.hidden_size) if shape.horizon > 1 else None
        )
        self.recurrent: nn.LSTM | nn.GRU | nn.RNN = parameters.cell.build(
            parameters.hidden_size, parameters.hidden_size, parameters.num_layers, parameters.dropout, False
        )
        self.conditioning: nn.Linear = nn.Linear(parameters.hidden_size, parameters.conditioning_size)
        self.denoiser: TimeGradDenoiser = TimeGradDenoiser(parameters, shape.output_channels)
        self.process: DiffusionProcess = parameters.noise_schedule.build_process()
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def compute_loss(self, past: Tensor, future_targets: Tensor) -> Tensor:
        expected: tuple[int, int, int] = (past.shape[0], self.horizon, self.output_channels)
        if tuple(future_targets.shape) != expected:
            raise ValueError(f"future_targets must have shape {expected}, got {tuple(future_targets.shape)}")
        encoded: Tensor
        state: RecurrentState
        encoded, state = self.recurrent(self.past_projection(past))
        # The hidden state after step t conditions the targets of step t + 1.
        hidden: Tensor = encoded
        if self.target_projection is not None:
            decoded: Tensor = self.recurrent(self.target_projection(future_targets[:, :-1, :]), state)[0]
            hidden = torch.cat([encoded, decoded], dim=1)
        targets: Tensor = torch.cat([past.index_select(2, self.target_channels)[:, 1:, :], future_targets], dim=1)
        conditioning: Tensor = self.conditioning(hidden).flatten(0, 1)
        clean: Tensor = targets.flatten(0, 1)
        noised: NoisedSample = self.process.corrupt(clean)
        predicted: Tensor = self.denoiser(noised.noisy, noised.steps, conditioning)
        return (predicted - noised.noise).square().mean()

    @override
    def sample(self, past: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        batch: int = past.shape[0]
        encoded: Tensor
        state: RecurrentState
        encoded, state = self.recurrent(self.past_projection(past))
        hidden: Tensor = encoded[:, -1:, :].repeat_interleave(num_samples, dim=0)
        state = self._repeat_state(state, num_samples)
        projection: nn.Linear | None = self.target_projection
        steps: list[Tensor] = []
        for position in range(self.horizon):
            drawn: Tensor = self._draw_step(self.conditioning(hidden[:, 0, :]), generator)
            steps.append(drawn)
            if projection is not None and position + 1 < self.horizon:
                hidden, state = self.recurrent(projection(drawn.unsqueeze(1)), state)
        return torch.stack(steps, dim=1).reshape(batch, num_samples, self.horizon, self.output_channels)

    def _draw_step(self, conditioning: Tensor, generator: torch.Generator | None) -> Tensor:
        def predict_noise(noisy: Tensor, diffusion_steps: Tensor) -> Tensor:
            return self.denoiser(noisy, diffusion_steps, conditioning)

        return self.process.sample(predict_noise, (conditioning.shape[0], self.output_channels), generator)

    @staticmethod
    def _repeat_state(state: RecurrentState, num_samples: int) -> RecurrentState:
        if isinstance(state, tuple):
            return (state[0].repeat_interleave(num_samples, dim=1), state[1].repeat_interleave(num_samples, dim=1))
        return state.repeat_interleave(num_samples, dim=1)
