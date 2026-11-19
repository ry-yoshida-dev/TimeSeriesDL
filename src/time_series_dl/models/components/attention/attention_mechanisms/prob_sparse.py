import math
from typing import override

import torch
from torch import Tensor, nn

from ..attention_mechanism import AttentionMechanism


class ProbSparseAttentionMechanism(AttentionMechanism):
    """
    ProbSparse self-attention of Informer (Zhou et al., AAAI 2021).

    Only the ``factor * ceil(ln L_Q)`` queries with the largest sparsity measurement, estimated on
    ``factor * ceil(ln L_K)`` randomly sampled keys, attend to every key. The other queries output the mean of the
    values, or their cumulative mean when causal.

    Parameters
    ----------
    factor : int
        Sampling factor c of the active queries and the sampled keys.
    is_causal : bool
        Whether each query sees only the keys up to its own position; queries and keys then have the same length.
    dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If factor is not positive or dropout is not in [0, 1).
    """

    def __init__(self, factor: int, is_causal: bool, dropout: float) -> None:
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.factor: int = factor
        self.is_causal: bool = is_causal
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def count_active_queries(self, query_length: int) -> int:
        """
        Get the number u of queries that attend to every key.

        Parameters
        ----------
        query_length : int
            Number of queries.

        Returns
        -------
        int
            ``factor * ceil(ln query_length)``, clipped to [1, query_length].
        """
        return self._count(query_length)

    def count_sampled_keys(self, key_length: int) -> int:
        """
        Get the number of keys sampled per query to estimate its sparsity.

        Parameters
        ----------
        key_length : int
            Number of keys.

        Returns
        -------
        int
            ``factor * ceil(ln key_length)``, clipped to [1, key_length].
        """
        return self._count(key_length)

    def _count(self, length: int) -> int:
        return max(1, min(self.factor * math.ceil(math.log(length)), length))

    @override
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        query_length: int = queries.shape[2]
        key_length: int = keys.shape[2]
        if self.is_causal and query_length != key_length:
            raise ValueError(f"causal attention needs as many queries as keys, got {query_length} and {key_length}")
        sample_index: Tensor = torch.randint(
            key_length, (query_length, self.count_sampled_keys(key_length)), device=queries.device
        )
        sampled_scores: Tensor = torch.einsum("bhqe,bhqse->bhqs", queries, keys[:, :, sample_index, :])
        sparsity: Tensor = sampled_scores.amax(dim=-1) - sampled_scores.sum(dim=-1) / key_length
        active: Tensor = sparsity.topk(self.count_active_queries(query_length), dim=-1, sorted=False).indices
        active_queries: Tensor = queries.gather(2, active.unsqueeze(-1).expand(-1, -1, -1, queries.shape[3]))
        scores: Tensor = active_queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[3])
        context: Tensor
        if self.is_causal:
            positions: Tensor = torch.arange(key_length, device=queries.device)
            scores = scores.masked_fill(positions > active.unsqueeze(-1), float("-inf"))
            counts: Tensor = torch.arange(1, key_length + 1, device=values.device, dtype=values.dtype)
            context = values.cumsum(dim=2) / counts.view(1, 1, -1, 1)
        else:
            context = values.mean(dim=2, keepdim=True).expand(-1, -1, query_length, -1)
        weights: Tensor = self.dropout(scores.softmax(dim=-1))
        updates: Tensor = weights @ values
        return context.scatter(2, active.unsqueeze(-1).expand(-1, -1, -1, values.shape[3]), updates)
