from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.graph import AdaptiveAdjacency
from ....interfaces import ForecastNetwork, ForecastShape
from .layer import GraphWaveNetLayer

if TYPE_CHECKING:
    from .parameters import GraphWaveNetParameters


class GraphWaveNetForecastNetwork(ForecastNetwork):
    """
    Graph WaveNet over every input channel as a node: a self-adaptive adjacency from two node embeddings, blocks of
    gated dilated convolutions and diffusion graph convolutions whose skip connections feed a pointwise output module
    that maps each target node to its whole horizon at once.

    Windows shorter than the receptive field are padded with zeros at the start; of longer windows, the output module
    reads the skip connections of the last step, as in the official code where the window matches the receptive
    field.

    Parameters
    ----------
    parameters : GraphWaveNetParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: GraphWaveNetParameters, shape: ForecastShape) -> None:
        super().__init__()
        dilations: list[int] = [2**index for index in range(parameters.num_layers)] * parameters.num_blocks
        receptive_field: int = 1 + (parameters.kernel_size - 1) * sum(dilations)
        self.padded_length: int = max(shape.input_length, receptive_field)
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.adjacency: AdaptiveAdjacency = AdaptiveAdjacency(
            shape.input_channels, parameters.embedding_size, is_symmetric=False
        )
        self.start: nn.Conv2d = nn.Conv2d(1, parameters.residual_channels, 1)
        self.layers: nn.ModuleList = nn.ModuleList(
            GraphWaveNetLayer(parameters, dilation, is_propagating=index < len(dilations) - 1)
            for index, dilation in enumerate(dilations)
        )
        self.end: nn.Conv2d = nn.Conv2d(parameters.skip_channels, parameters.end_channels, 1)
        self.projection: nn.Linear = nn.Linear(parameters.end_channels, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        inputs: Tensor = past.transpose(1, 2).unsqueeze(1)
        inputs = functional.pad(inputs, (self.padded_length - inputs.shape[-1], 0))
        # The official code propagates along the transposed adaptive adjacency.
        transition: Tensor = self.adjacency().transpose(0, 1)
        hidden: Tensor = self.start(inputs)
        # The skip connections are aligned at the end, and only their sum at the last step is read.
        last_skips: list[Tensor] = []
        for layer in self.layers:
            propagated: Tensor | None
            contribution: Tensor
            propagated, contribution = layer(hidden, transition)
            last_skips.append(contribution[..., -1:])
            if propagated is not None:
                hidden = propagated
        skip: Tensor = torch.stack(last_skips).sum(dim=0)
        features: Tensor = torch.relu(self.end(torch.relu(skip))).squeeze(-1)
        outputs: Tensor = self.projection(features.transpose(1, 2).index_select(1, self.target_channels))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
