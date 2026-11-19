from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class AdaptiveAdjacency(nn.Module):
    """
    Dense adjacency learned from node embeddings, ``softmax(relu(source @ target.T))`` normalized over each row.

    This is the self-adaptive adjacency of Graph WaveNet (two embeddings) and the data-adaptive graph of AGCRN (one
    embedding used as both source and target).

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    embedding_size : int
        Size of each node embedding.
    is_symmetric : bool
        Whether the source and target embeddings are the same, which makes the scores before the softmax symmetric.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, num_nodes: int, embedding_size: int, is_symmetric: bool) -> None:
        if num_nodes <= 0 or embedding_size <= 0:
            raise ValueError(f"num_nodes and embedding_size must be positive, got {num_nodes} and {embedding_size}")
        super().__init__()
        num_embeddings: int = 1 if is_symmetric else 2
        self.embeddings: nn.Parameter = nn.Parameter(torch.randn(num_embeddings, num_nodes, embedding_size))

    @property
    def node_embeddings(self) -> Tensor:
        """
        Get the source embedding of each node, which AGCRN also uses to generate node-specific weights.

        Returns
        -------
        Tensor
            Embeddings with shape (num_nodes, embedding_size).
        """
        return self.embeddings[0]

    def forward(self) -> Tensor:
        """
        Compute the adjacency.

        Returns
        -------
        Tensor
            Adjacency with shape (num_nodes, num_nodes) whose rows are non-negative and sum to one.
        """
        scores: Tensor = self.embeddings[0] @ self.embeddings[-1].transpose(0, 1)
        return functional.softmax(functional.relu(scores), dim=1)

    def __call__(self) -> Tensor:
        return cast(Tensor, super().__call__())
