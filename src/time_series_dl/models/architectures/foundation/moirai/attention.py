from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn

from .rotary_embedding import MoiraiRotaryEmbedding


class MoiraiAttention(nn.Module):
    """
    Any-variate attention of Moirai: multi-head attention with root-mean-square normalized queries and keys, a rotary
    embedding of the time index, and a learned bias per head for token pairs of the same and of different variates.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    num_heads : int
        Number of heads.
    dropout : float
        Dropout probability of the attention weights.
    epsilon : float
        Epsilon of the query and key normalizations.
    """

    def __init__(self, model_size: int, num_heads: int, dropout: float, epsilon: float) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_size: int = model_size // num_heads
        self.dropout: float = dropout
        self.query: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.key: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.value: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.output: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.query_norm: nn.RMSNorm = nn.RMSNorm(self.head_size, eps=epsilon)
        self.key_norm: nn.RMSNorm = nn.RMSNorm(self.head_size, eps=epsilon)
        # Row 0: bias between tokens of different variates; row 1: of the same variate.
        self.variate_bias: nn.Embedding = nn.Embedding(2, num_heads)

    def forward(self, hidden: Tensor, variate_ids: Tensor, time_ids: Tensor, rotary: MoiraiRotaryEmbedding) -> Tensor:
        """
        Attend from every token to every token.

        Parameters
        ----------
        hidden : Tensor
            Token representations with shape (batch, tokens, model_size).
        variate_ids : Tensor
            Variate index of each token with shape (batch, tokens).
        time_ids : Tensor
            Time index of each token with shape (batch, tokens).
        rotary : MoiraiRotaryEmbedding
            The rotary embedding shared by every layer.

        Returns
        -------
        Tensor
            The attended representations with shape (batch, tokens, model_size).
        """
        query: Tensor = rotary.apply(self.query_norm(self._split_heads(self.query(hidden))), time_ids)
        key: Tensor = rotary.apply(self.key_norm(self._split_heads(self.key(hidden))), time_ids)
        value: Tensor = self._split_heads(self.value(hidden))
        is_same_variate: Tensor = (variate_ids[:, :, None] == variate_ids[:, None, :])[:, None]
        different: Tensor = self.variate_bias.weight[0][None, :, None, None]
        same: Tensor = self.variate_bias.weight[1][None, :, None, None]
        bias: Tensor = ~is_same_variate * different + is_same_variate * same
        attended: Tensor = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias.to(query.dtype),
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.head_size**-0.5,
        )
        projected: Tensor = self.output(attended.transpose(1, 2).flatten(start_dim=2))
        return projected

    def __call__(self, hidden: Tensor, variate_ids: Tensor, time_ids: Tensor, rotary: MoiraiRotaryEmbedding) -> Tensor:
        return cast(Tensor, super().__call__(hidden, variate_ids, time_ids, rotary))

    def _split_heads(self, projected: Tensor) -> Tensor:
        batch, length, _ = projected.shape
        split: Tensor = projected.reshape(batch, length, self.num_heads, self.head_size).transpose(1, 2)
        return split
