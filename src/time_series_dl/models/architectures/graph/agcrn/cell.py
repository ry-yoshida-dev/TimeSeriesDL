from typing import cast

import torch
from torch import Tensor, nn

from ....components.graph import NodeAdaptiveGraphConvolution


class AGCRNCell(nn.Module):
    """
    Graph convolutional gated recurrent unit of AGCRN, whose linear maps are node adaptive graph convolutions.

    Parameters
    ----------
    input_size : int
        Number of input features per node.
    hidden_size : int
        Size of the hidden state of each node.
    num_supports : int
        Number of propagation matrices of the graph convolutions.
    embedding_size : int
        Size of the node embeddings.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, input_size: int, hidden_size: int, num_supports: int, embedding_size: int) -> None:
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError(f"input_size and hidden_size must be positive, got {input_size} and {hidden_size}")
        super().__init__()
        self.hidden_size: int = hidden_size
        self.gates: NodeAdaptiveGraphConvolution = NodeAdaptiveGraphConvolution(
            input_size + hidden_size, 2 * hidden_size, num_supports, embedding_size
        )
        self.candidate: NodeAdaptiveGraphConvolution = NodeAdaptiveGraphConvolution(
            input_size + hidden_size, hidden_size, num_supports, embedding_size
        )

    def forward(self, inputs: Tensor, state: Tensor, supports: Tensor, node_embeddings: Tensor) -> Tensor:
        """
        Advance the hidden state by one step.

        Parameters
        ----------
        inputs : Tensor
            Inputs with shape (batch, nodes, input_size).
        state : Tensor
            Hidden state with shape (batch, nodes, hidden_size).
        supports : Tensor
            Propagation matrices with shape (num_supports, nodes, nodes).
        node_embeddings : Tensor
            Embeddings with shape (nodes, embedding_size).

        Returns
        -------
        Tensor
            The next hidden state with shape (batch, nodes, hidden_size).
        """
        gates: Tensor = torch.sigmoid(self.gates(torch.cat([inputs, state], dim=-1), supports, node_embeddings))
        reset: Tensor = gates[..., : self.hidden_size]
        update: Tensor = gates[..., self.hidden_size :]
        candidate: Tensor = torch.tanh(
            self.candidate(torch.cat([inputs, reset * state], dim=-1), supports, node_embeddings)
        )
        return update * state + (1.0 - update) * candidate

    def __call__(self, inputs: Tensor, state: Tensor, supports: Tensor, node_embeddings: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(inputs, state, supports, node_embeddings))
