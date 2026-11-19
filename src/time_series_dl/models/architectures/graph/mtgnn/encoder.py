from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.graph import DirectedGraphLearner
from .dilated_inception import MTGNNDilatedInception
from .layer import MTGNNLayer

if TYPE_CHECKING:
    from .parameters import MTGNNParameters


class MTGNNEncoder(nn.Module):
    """
    MTGNN up to its output module: a graph learned from node embeddings, gated dilated inception convolutions and
    mix-hop propagation layers whose skip connections sum to one feature vector per node.

    Windows shorter than the receptive field are padded with zeros at the start, as in the official code.

    Parameters
    ----------
    parameters : MTGNNParameters
        The model configuration.
    num_nodes : int
        Number of nodes, the channels of the series.
    input_length : int
        Number of time steps of the input.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, parameters: MTGNNParameters, num_nodes: int, input_length: int) -> None:
        if num_nodes <= 0 or input_length <= 0:
            raise ValueError(f"num_nodes and input_length must be positive, got {num_nodes} and {input_length}")
        super().__init__()
        span: int = max(MTGNNDilatedInception.KERNEL_SIZES) - 1
        dilations: list[int] = [parameters.dilation_base**index for index in range(parameters.num_layers)]
        receptive_field: int = 1 + span * sum(dilations)
        self.padded_length: int = max(input_length, receptive_field)
        self.graph_learner: DirectedGraphLearner = DirectedGraphLearner(
            num_nodes, parameters.embedding_size, parameters.num_neighbors, parameters.saturation
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.start: nn.Conv2d = nn.Conv2d(1, parameters.residual_channels, 1)
        self.input_skip: nn.Conv2d = nn.Conv2d(1, parameters.skip_channels, (1, self.padded_length))
        self.layers: nn.ModuleList = nn.ModuleList(
            MTGNNLayer(parameters, num_nodes, dilation, self.padded_length - span * sum(dilations[: index + 1]))
            for index, dilation in enumerate(dilations)
        )
        self.output_skip: nn.Conv2d = nn.Conv2d(
            parameters.residual_channels, parameters.skip_channels, (1, self.padded_length - receptive_field + 1)
        )
        self.end: nn.Conv2d = nn.Conv2d(parameters.skip_channels, parameters.end_channels, 1)

    def forward(self, values: Tensor) -> Tensor:
        """
        Encode every node of a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, nodes).

        Returns
        -------
        Tensor
            Node features with shape (batch, nodes, end_channels).
        """
        inputs: Tensor = values.transpose(1, 2).unsqueeze(1)
        inputs = functional.pad(inputs, (self.padded_length - inputs.shape[-1], 0))
        adjacency: Tensor = self.graph_learner()
        skip: Tensor = self.input_skip(self.dropout(inputs))
        hidden: Tensor = self.start(inputs)
        for layer in self.layers:
            contribution: Tensor
            hidden, contribution = layer(hidden, adjacency)
            skip = skip + contribution
        skip = skip + self.output_skip(hidden)
        features: Tensor = torch.relu(self.end(torch.relu(skip)))
        return features.squeeze(-1).transpose(1, 2)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
