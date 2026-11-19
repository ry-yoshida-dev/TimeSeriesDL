import math
from typing import cast

import torch
from torch import Tensor, nn


class SAITSDiagonallyMaskedAttention(nn.Module):
    """
    Multi-head self-attention in which no time step attends to itself (Du et al., 2023), so each step's value is
    estimated only from the other steps.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads.
    key_size : int
        Size of the queries and keys of each head.
    value_size : int
        Size of the values of each head.
    attention_dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If a size is not positive or attention_dropout is not in [0, 1).
    """

    def __init__(
        self, model_size: int, num_heads: int, key_size: int, value_size: int, attention_dropout: float
    ) -> None:
        if min(model_size, num_heads, key_size, value_size) <= 0:
            raise ValueError(
                "model_size, num_heads, key_size and value_size must be positive, got "
                + f"{model_size}, {num_heads}, {key_size} and {value_size}"
            )
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {attention_dropout}")
        super().__init__()
        self.num_heads: int = num_heads
        self.key_size: int = key_size
        self.query_projection: nn.Linear = nn.Linear(model_size, num_heads * key_size, bias=False)
        self.key_projection: nn.Linear = nn.Linear(model_size, num_heads * key_size, bias=False)
        self.value_projection: nn.Linear = nn.Linear(model_size, num_heads * value_size, bias=False)
        self.output_projection: nn.Linear = nn.Linear(num_heads * value_size, model_size, bias=False)
        self.dropout: nn.Dropout = nn.Dropout(attention_dropout)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Attend from every time step to every other time step.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size); the length must exceed one.

        Returns
        -------
        tuple[Tensor, Tensor]
            The output tokens with shape (batch, length, model_size), and the attention weights with shape
            (batch, num_heads, length, length).

        Raises
        ------
        ValueError
            If the sequence has a single time step, which has nothing else to attend to.
        """
        length: int = tokens.shape[1]
        if length < 2:
            raise ValueError(f"diagonally masked attention needs at least two time steps, got {length}")
        queries: Tensor = self._split(self.query_projection(tokens))
        keys: Tensor = self._split(self.key_projection(tokens))
        values: Tensor = self._split(self.value_projection(tokens))
        scores: Tensor = queries @ keys.transpose(-2, -1) / math.sqrt(self.key_size)
        is_diagonal: Tensor = torch.eye(length, dtype=torch.bool, device=tokens.device)
        weights: Tensor = torch.softmax(scores.masked_fill(is_diagonal, -math.inf), dim=-1)
        attended: Tensor = self.dropout(weights) @ values
        outputs: Tensor = self.output_projection(attended.transpose(1, 2).flatten(start_dim=2))
        return outputs, weights

    def _split(self, tokens: Tensor) -> Tensor:
        return tokens.reshape(tokens.shape[0], tokens.shape[1], self.num_heads, -1).transpose(1, 2)

    def __call__(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(tokens))
