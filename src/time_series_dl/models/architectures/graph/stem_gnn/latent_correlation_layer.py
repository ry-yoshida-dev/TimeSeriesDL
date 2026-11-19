from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.graph import ChebyshevPolynomials

if TYPE_CHECKING:
    from .parameters import StemGNNParameters


class StemGNNLatentCorrelationLayer(nn.Module):
    """
    Latent correlation layer of StemGNN: a graph inferred from the window itself.

    A gated recurrent unit reads each node's series and its last hidden state represents the node; additive
    self-attention ``softmax_j(leaky_relu(k_i + q_j))`` between the nodes, averaged over the batch and symmetrized,
    is the adjacency ``W``. The normalized Laplacian ``L = D^(-1/2) (D - W) D^(-1/2)`` is expanded into Chebyshev
    polynomials, which the StemGNN blocks use as graph Fourier filters.

    Parameters
    ----------
    parameters : StemGNNParameters
        The model configuration.
    """

    def __init__(self, parameters: StemGNNParameters) -> None:
        super().__init__()
        self.recurrent: nn.GRU = nn.GRU(1, parameters.latent_size, batch_first=True)
        self.key: nn.Parameter = nn.Parameter(torch.empty(parameters.latent_size, 1))
        self.query: nn.Parameter = nn.Parameter(torch.empty(parameters.latent_size, 1))
        _ = nn.init.xavier_uniform_(self.key, gain=1.414)
        _ = nn.init.xavier_uniform_(self.query, gain=1.414)
        self.negative_slope: float = parameters.attention_negative_slope
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.polynomials: ChebyshevPolynomials = ChebyshevPolynomials(num_terms=parameters.num_chebyshev_terms)

    def forward(self, values: Tensor) -> Tensor:
        """
        Infer the graph of a batch of windows.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, nodes, length).

        Returns
        -------
        Tensor
            Chebyshev polynomials of the normalized Laplacian with shape (num_chebyshev_terms, nodes, nodes).
        """
        batch_size: int = values.shape[0]
        num_nodes: int = values.shape[1]
        hidden: Tensor
        _, hidden = self.recurrent(values.reshape(batch_size * num_nodes, -1, 1))
        representations: Tensor = hidden[-1].reshape(batch_size, num_nodes, -1)
        scores: Tensor = (representations @ self.key) + (representations @ self.query).transpose(1, 2)
        attention: Tensor = self.dropout(
            functional.softmax(functional.leaky_relu(scores, self.negative_slope), dim=2)
        ).mean(dim=0)
        adjacency: Tensor = 0.5 * (attention + attention.transpose(0, 1))
        degree: Tensor = adjacency.sum(dim=1)
        inverse_root_degree: Tensor = 1.0 / (degree.sqrt() + 1e-7)
        laplacian: Tensor = (
            inverse_root_degree[:, None] * (torch.diag(degree) - adjacency) * inverse_root_degree[None, :]
        )
        return self.polynomials.evaluate(laplacian)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
