from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.graph import AdaptiveAdjacency, ChebyshevPolynomials
from ....interfaces import ForecastNetwork, ForecastShape
from .cell import AGCRNCell

if TYPE_CHECKING:
    from .parameters import AGCRNParameters


class AGCRNForecastNetwork(ForecastNetwork):
    """
    AGCRN over every input channel as a node: a data adaptive graph from one node embedding, stacked graph
    convolutional recurrent layers over the window, and a pointwise map of each target node's last hidden state to
    its whole horizon at once.

    Parameters
    ----------
    parameters : AGCRNParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: AGCRNParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.hidden_size: int = parameters.hidden_size
        self.adjacency: AdaptiveAdjacency = AdaptiveAdjacency(
            shape.input_channels, parameters.embedding_size, is_symmetric=True
        )
        self.polynomials: ChebyshevPolynomials = ChebyshevPolynomials(num_terms=parameters.num_chebyshev_terms)
        self.cells: nn.ModuleList = nn.ModuleList(
            AGCRNCell(
                1 if index == 0 else parameters.hidden_size,
                parameters.hidden_size,
                parameters.num_chebyshev_terms,
                parameters.embedding_size,
            )
            for index in range(parameters.num_layers)
        )
        self.projection: nn.Linear = nn.Linear(parameters.hidden_size, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        node_embeddings: Tensor = self.adjacency.node_embeddings
        supports: Tensor = self.polynomials.evaluate(self.adjacency())
        sequence: Tensor = past.unsqueeze(-1)
        state: Tensor = past.new_zeros(past.shape[0], past.shape[2], self.hidden_size)
        for cell in self.cells:
            state = past.new_zeros(past.shape[0], past.shape[2], self.hidden_size)
            states: list[Tensor] = []
            for step in range(sequence.shape[1]):
                state = cell(sequence[:, step], state, supports, node_embeddings)
                states.append(state)
            sequence = torch.stack(states, dim=1)
        outputs: Tensor = self.projection(state.index_select(1, self.target_channels))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
