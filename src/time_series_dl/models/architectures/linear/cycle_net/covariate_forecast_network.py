from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import CovariateForecastNetwork, CovariateForecastShape, CovariateTensors
from .recurrent_cycle import CycleNetRecurrentCycle

if TYPE_CHECKING:
    from .parameters import CycleNetParameters


class CycleNetCovariateForecastNetwork(CovariateForecastNetwork):
    """
    Residual cycle forecasting: the learnable cycle at the phase of each step is removed from each target channel's
    window, the backbone shared by the channels maps the residual window to the residual horizon, and the cycle at the
    phase of each horizon step is added back.

    The phase of every step of the window and the horizon is read from the known covariate channel
    ``parameters.phase_channel``, which must hold integers in ``[0, cycle_length)``, unscaled. The other covariates
    and the past-observed channels that are not targets are not read. The cycle is added only to the distribution
    parameters in the scale of the values (``ForecastShape.level_parameters``).

    Parameters
    ----------
    parameters : CycleNetParameters
        The model configuration.
    shape : CovariateForecastShape
        The dimensions of the forecast and of the covariates.

    Raises
    ------
    ValueError
        If the known covariates have no channel ``parameters.phase_channel``.
    """

    def __init__(self, parameters: CycleNetParameters, shape: CovariateForecastShape) -> None:
        if parameters.phase_channel >= shape.known_channels:
            raise ValueError(
                f"CycleNet reads the phase from known covariate channel {parameters.phase_channel}, but there are "
                + f"{shape.known_channels} known covariate channels"
            )
        super().__init__()
        forecast = shape.forecast
        self.cycle_length: int = parameters.cycle_length
        self.phase_channel: int = parameters.phase_channel
        self.input_length: int = forecast.input_length
        self.horizon: int = forecast.horizon
        self.num_parameters: int = forecast.num_parameters
        self.recurrent_cycle: CycleNetRecurrentCycle = CycleNetRecurrentCycle(
            parameters.cycle_length, forecast.output_channels
        )
        self.backbone: nn.Module = parameters.backbone.build(
            forecast.input_length, parameters.hidden_size, forecast.horizon * forecast.num_parameters
        )
        level_mask: Tensor = torch.zeros(forecast.num_parameters)
        level_mask[list(forecast.level_parameters)] = 1.0
        self.level_mask: Tensor
        self.register_buffer("level_mask", level_mask, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(forecast.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, covariates: CovariateTensors, future_targets: Tensor | None = None) -> Tensor:
        phase: Tensor = self._read_phase(covariates.known)
        cycle: Tensor = self.recurrent_cycle(phase)
        series: Tensor = past.index_select(2, self.target_channels)
        residual: Tensor = (series - cycle[:, : self.input_length]).transpose(1, 2)
        mapped: Tensor = self.backbone(residual)
        batch_size: int = past.shape[0]
        outputs: Tensor = mapped.reshape(batch_size, -1, self.horizon, self.num_parameters).transpose(1, 2)
        return outputs + cycle[:, self.input_length :].unsqueeze(-1) * self.level_mask

    def _read_phase(self, known: Tensor) -> Tensor:
        values: Tensor = known[:, :, self.phase_channel]
        phase: Tensor = values.round().long()
        is_integral: bool = bool(torch.equal(phase.to(values.dtype), values))
        if not is_integral or bool(phase.min() < 0) or bool(phase.max() >= self.cycle_length):
            raise ValueError(
                f"known covariate channel {self.phase_channel} must hold the phase of each step as an integer in "
                + f"[0, {self.cycle_length}); pass it unscaled (covariate_scaling=NoScalingParameters())"
            )
        return phase
