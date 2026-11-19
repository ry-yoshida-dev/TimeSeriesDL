from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .fourier_graph_operators import FourierGNNGraphOperators

if TYPE_CHECKING:
    from .parameters import FourierGNNParameters


class FourierGNNForecastNetwork(ForecastNetwork):
    """
    FourierGNN over every input channel: each (channel, step) value is a node of a fully connected hypervariate
    graph, embedded by a learned vector; Fourier graph operators act on the discrete Fourier transform taken over all
    nodes, and after the inverse transform each target channel's steps are projected to a few features and mapped by
    a feed-forward network to its whole horizon at once.

    Parameters
    ----------
    parameters : FourierGNNParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    FIRST_HIDDEN_SIZE: int = 64

    def __init__(self, parameters: FourierGNNParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.token_embedding: nn.Parameter = nn.Parameter(torch.randn(parameters.embedding_size))
        self.operators: FourierGNNGraphOperators = FourierGNNGraphOperators(parameters)
        self.time_projection: nn.Parameter = nn.Parameter(
            torch.randn(shape.input_length, parameters.time_projection_size)
        )
        self.output: nn.Sequential = nn.Sequential(
            nn.Linear(parameters.embedding_size * parameters.time_projection_size, self.FIRST_HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(self.FIRST_HIDDEN_SIZE, parameters.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(parameters.hidden_size, shape.horizon * shape.num_parameters),
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch_size: int = past.shape[0]
        length: int = past.shape[1]
        num_channels: int = past.shape[2]
        num_nodes: int = num_channels * length
        tokens: Tensor = past.transpose(1, 2).reshape(batch_size, num_nodes, 1) * self.token_embedding
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(tokens, dim=1, norm="ortho"))
        filtered: Tensor = self.operators(spectrum)
        hidden: Tensor = cast(Tensor, torch.fft.irfft(filtered, n=num_nodes, dim=1, norm="ortho"))
        per_channel: Tensor = hidden.reshape(batch_size, num_channels, length, -1).index_select(1, self.target_channels)
        projected: Tensor = per_channel.transpose(2, 3) @ self.time_projection
        outputs: Tensor = self.output(projected.flatten(start_dim=2))
        return outputs.reshape(batch_size, -1, self.horizon, self.num_parameters).transpose(1, 2)
