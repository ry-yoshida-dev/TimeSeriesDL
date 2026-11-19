from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .frequency_layer import FreTSFrequencyLayer

if TYPE_CHECKING:
    from .parameters import FreTSParameters


class FreTSForecastNetwork(ForecastNetwork):
    """
    Each value is multiplied by a learned embedding vector, the embeddings are mixed across channels and along time by
    frequency-domain MLPs with a residual connection, and each target channel's embedded window is mapped to its
    horizon by an MLP shared across channels.

    Parameters
    ----------
    parameters : FreTSParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: FreTSParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.embedding: nn.Parameter = nn.Parameter(torch.randn(parameters.embed_size))
        self.channel_learner: FreTSFrequencyLayer | None = (
            FreTSFrequencyLayer(parameters.embed_size, parameters.sparsity_threshold, parameters.weight_scale)
            if parameters.is_channel_mixing
            else None
        )
        self.temporal_learner: FreTSFrequencyLayer = FreTSFrequencyLayer(
            parameters.embed_size, parameters.sparsity_threshold, parameters.weight_scale
        )
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(shape.input_length * parameters.embed_size, parameters.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(parameters.hidden_size, shape.horizon * shape.num_parameters),
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        tokens: Tensor = past.transpose(1, 2).unsqueeze(-1) * self.embedding
        mixed: Tensor = tokens
        if self.channel_learner is not None:
            mixed = self.channel_learner(mixed, 1)
        mixed = self.temporal_learner(mixed, 2) + tokens
        selected: Tensor = mixed.index_select(1, self.target_channels)
        outputs: Tensor = self.projection(selected.flatten(start_dim=2))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
