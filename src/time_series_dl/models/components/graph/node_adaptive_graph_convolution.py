from typing import cast

import torch
from torch import Tensor, nn


class NodeAdaptiveGraphConvolution(nn.Module):
    """
    Graph convolution with node adaptive parameter learning (AGCRN).

    Instead of one weight shared by all nodes, every node gets its own weights and bias as a linear combination of
    shared pools, ``W_n = E_n W_pool`` and ``b_n = E_n b_pool``, so the parameters grow with the embedding size rather
    than with the number of nodes. The node features propagated by each support are mapped by these weights and
    summed.

    Parameters
    ----------
    input_size : int
        Number of input features per node.
    output_size : int
        Number of output features per node.
    num_supports : int
        Number of propagation matrices, such as Chebyshev polynomials of the adjacency.
    embedding_size : int
        Size of the node embeddings the weights are generated from.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, input_size: int, output_size: int, num_supports: int, embedding_size: int) -> None:
        if input_size <= 0 or output_size <= 0 or num_supports <= 0 or embedding_size <= 0:
            raise ValueError(
                "input_size, output_size, num_supports and embedding_size must be positive, got "
                + f"{input_size}, {output_size}, {num_supports} and {embedding_size}"
            )
        super().__init__()
        self.weight_pool: nn.Parameter = nn.Parameter(
            torch.empty(embedding_size, num_supports, input_size, output_size)
        )
        self.bias_pool: nn.Parameter = nn.Parameter(torch.empty(embedding_size, output_size))
        # The official training script initializes every parameter with more than one dimension this way.
        _ = nn.init.xavier_uniform_(self.weight_pool)
        _ = nn.init.xavier_uniform_(self.bias_pool)

    def forward(self, features: Tensor, supports: Tensor, node_embeddings: Tensor) -> Tensor:
        """
        Convolve node features over the graph.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, nodes, input_size).
        supports : Tensor
            Propagation matrices with shape (num_supports, nodes, nodes); entry (i, j) weighs the message node i
            receives from node j.
        node_embeddings : Tensor
            Embeddings with shape (nodes, embedding_size).

        Returns
        -------
        Tensor
            Features with shape (batch, nodes, output_size).
        """
        weights: Tensor = torch.einsum("nd,dkio->nkio", node_embeddings, self.weight_pool)
        bias: Tensor = node_embeddings @ self.bias_pool
        propagated: Tensor = torch.einsum("knm,bmc->bnkc", supports, features)
        return torch.einsum("bnki,nkio->bno", propagated, weights) + bias

    def __call__(self, features: Tensor, supports: Tensor, node_embeddings: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(features, supports, node_embeddings))
