from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.normalization import InstanceStatistics, ReversibleInstanceNormalization
from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import TimeMixerPlusPlusEncoder

if TYPE_CHECKING:
    from .parameters import TimeMixerPlusPlusParameters


class TimeMixerPlusPlusForecastNetwork(ForecastNetwork):
    """
    Reversible instance normalization, the multi-scale encoder, and one predictor per scale mapping its tokens along
    time to the horizon; the predictions are summed and projected to the parameters of every channel, of which the
    target channels are selected.

    Only the parameters in the scale of the values are denormalized.

    Parameters
    ----------
    parameters : TimeMixerPlusPlusParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If the input is too short for the down-samplings.
    """

    def __init__(self, parameters: TimeMixerPlusPlusParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.normalization: ReversibleInstanceNormalization = ReversibleInstanceNormalization(shape.input_channels)
        self.encoder: TimeMixerPlusPlusEncoder = TimeMixerPlusPlusEncoder(
            parameters, shape.input_length, shape.input_channels
        )
        self.predictors: nn.ModuleList = nn.ModuleList(
            nn.Linear(length, shape.horizon) for length in self.encoder.lengths
        )
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.input_channels * shape.num_parameters)
        is_level: Tensor = torch.zeros(shape.num_parameters, dtype=torch.bool)
        is_level[list(shape.level_parameters)] = True
        self.is_level: Tensor
        self.register_buffer("is_level", is_level, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        statistics: InstanceStatistics = self.normalization.compute_statistics(past)
        tokens: list[Tensor] = self.encoder(self.normalization.normalize(past, statistics))
        predictions: list[Tensor] = [
            predictor(sequence.transpose(1, 2)).transpose(1, 2)
            for predictor, sequence in zip(self.predictors, tokens, strict=True)
        ]
        projected: Tensor = self.projection(torch.stack(predictions, dim=0).sum(dim=0))
        outputs: Tensor = projected.reshape(past.shape[0], self.horizon, -1, self.num_parameters)
        outputs = torch.where(self.is_level, self.normalization.denormalize(outputs, statistics), outputs)
        return outputs.index_select(2, self.target_channels)
