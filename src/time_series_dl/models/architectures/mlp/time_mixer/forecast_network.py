from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor, nn

from ....components.normalization import InstanceStatistics, ReversibleInstanceNormalization
from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import TimeMixerEncoder

if TYPE_CHECKING:
    from .parameters import TimeMixerParameters


class TimeMixerForecastNetwork(ForecastNetwork):
    """
    Each target channel is down-sampled into several scales, each normalized by its own reversible instance
    normalization, encoded independently of the other channels, and every scale predicts the horizon; the predictions
    are summed (future multipredictor mixing).

    Parameters
    ----------
    parameters : TimeMixerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If the coarsest scale would have no time step.
    """

    def __init__(self, parameters: TimeMixerParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.encoder: TimeMixerEncoder = TimeMixerEncoder(
            parameters, shape.input_length, 1, is_channel_mixing=False, is_finest_output_only=False
        )
        self.normalizations: nn.ModuleList = nn.ModuleList(
            ReversibleInstanceNormalization(shape.output_channels, parameters.is_affine) for _ in self.encoder.lengths
        )
        self.predictors: nn.ModuleList = nn.ModuleList(
            nn.Linear(length, shape.horizon) for length in self.encoder.lengths
        )
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.num_parameters)
        is_level: Tensor = torch.zeros(shape.num_parameters, dtype=torch.bool)
        is_level[list(shape.level_parameters)] = True
        self.is_level: Tensor
        self.register_buffer("is_level", is_level, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch_size: int = past.shape[0]
        targets: Tensor = past.index_select(2, self.target_channels)
        num_channels: int = targets.shape[2]
        statistics: list[InstanceStatistics] = []
        scales: list[Tensor] = []
        for module, scale in zip(self.normalizations, self.encoder.downsample(targets), strict=True):
            normalization: ReversibleInstanceNormalization = cast(ReversibleInstanceNormalization, module)
            scale_statistics: InstanceStatistics = normalization.compute_statistics(scale)
            normalized: Tensor = normalization.normalize(scale, scale_statistics)
            statistics.append(scale_statistics)
            scales.append(normalized.transpose(1, 2).reshape(batch_size * num_channels, -1, 1))
        tokens: list[Tensor] = self.encoder(scales)
        predicted: Tensor = torch.stack(
            [
                self.projection(predictor(scale_tokens.transpose(1, 2)).transpose(1, 2))
                for predictor, scale_tokens in zip(self.predictors, tokens, strict=True)
            ]
        ).sum(dim=0)
        outputs: Tensor = predicted.reshape(batch_size, num_channels, self.horizon, self.num_parameters).transpose(1, 2)
        finest: ReversibleInstanceNormalization = cast(ReversibleInstanceNormalization, self.normalizations[0])
        return torch.where(self.is_level, finest.denormalize(outputs, statistics[0]), outputs)
