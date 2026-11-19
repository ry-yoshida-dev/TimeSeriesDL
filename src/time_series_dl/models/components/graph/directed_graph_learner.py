from typing import cast

import torch
from torch import Tensor, nn


class DirectedGraphLearner(nn.Module):
    """
    Graph learning layer of MTGNN: a sparse, uni-directed adjacency learned from two node embeddings.

    With ``M1 = tanh(saturation * Linear1(E1))`` and ``M2 = tanh(saturation * Linear2(E2))``, the adjacency is
    ``relu(tanh(saturation * (M1 @ M2.T - M2 @ M1.T))``; the antisymmetric scores give each pair of nodes an edge in
    at most one direction. Only the ``num_neighbors`` largest entries of each row are kept.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    embedding_size : int
        Size of each node embedding.
    num_neighbors : int
        Number of entries kept per row; all are kept when it is at least num_nodes.
    saturation : float
        Rate of the hyperbolic tangents, which controls how saturated the scores are.

    Raises
    ------
    ValueError
        If a size or the saturation is not positive.
    """

    def __init__(self, num_nodes: int, embedding_size: int, num_neighbors: int, saturation: float) -> None:
        if num_nodes <= 0 or embedding_size <= 0 or num_neighbors <= 0:
            raise ValueError(
                "num_nodes, embedding_size and num_neighbors must be positive, got "
                + f"{num_nodes}, {embedding_size} and {num_neighbors}"
            )
        if saturation <= 0.0:
            raise ValueError(f"saturation must be positive, got {saturation}")
        super().__init__()
        self.num_neighbors: int = min(num_neighbors, num_nodes)
        self.saturation: float = saturation
        self.source_embeddings: nn.Parameter = nn.Parameter(torch.empty(num_nodes, embedding_size))
        self.target_embeddings: nn.Parameter = nn.Parameter(torch.empty(num_nodes, embedding_size))
        # nn.Embedding initializes from the standard normal distribution.
        _ = nn.init.normal_(self.source_embeddings)
        _ = nn.init.normal_(self.target_embeddings)
        self.source_projection: nn.Linear = nn.Linear(embedding_size, embedding_size)
        self.target_projection: nn.Linear = nn.Linear(embedding_size, embedding_size)

    def forward(self) -> Tensor:
        """
        Compute the adjacency.

        While training, a little uniform noise breaks ties among the largest entries, as in the official code.

        Returns
        -------
        Tensor
            Non-negative adjacency with shape (num_nodes, num_nodes) and at most num_neighbors non-zero entries per
            row; entry (i, j) weighs the message node i receives from node j.
        """
        source: Tensor = torch.tanh(self.saturation * self.source_projection(self.source_embeddings))
        target: Tensor = torch.tanh(self.saturation * self.target_projection(self.target_embeddings))
        scores: Tensor = source @ target.transpose(0, 1) - target @ source.transpose(0, 1)
        adjacency: Tensor = torch.relu(torch.tanh(self.saturation * scores))
        ranked: Tensor = adjacency.detach()
        if self.training:
            ranked = ranked + torch.rand_like(ranked) * 0.01
        neighbors: Tensor = ranked.topk(self.num_neighbors, dim=1).indices
        mask: Tensor = torch.zeros_like(adjacency).scatter_(1, neighbors, 1.0)
        return adjacency * mask

    def __call__(self) -> Tensor:
        return cast(Tensor, super().__call__())
