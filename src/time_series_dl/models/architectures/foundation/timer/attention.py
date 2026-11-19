from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn

from .rotary_embedding import TimerRotaryEmbedding


class TimerAttention(nn.Module):
    """
    Causal multi-head self-attention of Timer with rotary position embeddings, biased query, key and value
    projections and an output projection without bias.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    num_heads : int
        Number of heads; must divide model_size.
    rotary_base : float
        Base of the rotary frequencies.
    dropout : float
        Dropout probability of the attention weights while training.
    """

    def __init__(self, model_size: int, num_heads: int, rotary_base: float, dropout: float) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.query: nn.Linear = nn.Linear(model_size, model_size)
        self.key: nn.Linear = nn.Linear(model_size, model_size)
        self.value: nn.Linear = nn.Linear(model_size, model_size)
        self.output: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.rotary_embedding: TimerRotaryEmbedding = TimerRotaryEmbedding(model_size // num_heads, rotary_base)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Attend from every position to itself and the positions before it.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            The attended values with the same shape.
        """
        query: Tensor = self.rotary_embedding(self._split_heads(self.query(hidden)))
        key: Tensor = self.rotary_embedding(self._split_heads(self.key(hidden)))
        value: Tensor = self._split_heads(self.value(hidden))
        attended: Tensor = functional.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        outputs: Tensor = self.output(attended.transpose(1, 2).flatten(start_dim=2))
        return outputs

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))

    def _split_heads(self, projected: Tensor) -> Tensor:
        batch, length, size = projected.shape
        split: Tensor = projected.reshape(batch, length, self.num_heads, size // self.num_heads).transpose(1, 2)
        return split
