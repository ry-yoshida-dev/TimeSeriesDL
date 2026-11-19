import math
from typing import cast

import torch
from torch import Tensor, nn


class AnomalyAttention(nn.Module):
    """
    Multi-head anomaly-attention: the series association is the softmax attention of the step tokens, and the prior
    association a Gaussian kernel of the distance between steps with a learned scale sigma per step and head,
    rescaled so every row sums to one.

    Parameters
    ----------
    model_size : int
        Size of each token.
    num_heads : int
        Number of heads; must divide model_size.
    length : int
        Number of steps of the windows.
    dropout : float
        Dropout probability of the series association.

    Raises
    ------
    ValueError
        If num_heads does not divide model_size or length is not positive.
    """

    def __init__(self, model_size: int, num_heads: int, length: int, dropout: float) -> None:
        if model_size % num_heads != 0:
            raise ValueError(f"num_heads {num_heads} must divide model_size {model_size}")
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        super().__init__()
        self.num_heads: int = num_heads
        self.query: nn.Linear = nn.Linear(model_size, model_size)
        self.key: nn.Linear = nn.Linear(model_size, model_size)
        self.value: nn.Linear = nn.Linear(model_size, model_size)
        self.sigma: nn.Linear = nn.Linear(model_size, num_heads)
        self.output: nn.Linear = nn.Linear(model_size, model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        steps: Tensor = torch.arange(length, dtype=torch.float32)
        self.distances: Tensor
        self.register_buffer("distances", (steps[:, None] - steps[None, :]).abs(), persistent=False)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Attend over the steps of a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            The attended tokens with shape (batch, length, model_size), and the series and the row-normalized prior
            associations, each with shape (batch, num_heads, length, length).
        """
        batch: int = tokens.shape[0]
        length: int = tokens.shape[1]
        queries: Tensor = self.query(tokens).view(batch, length, self.num_heads, -1).transpose(1, 2)
        keys: Tensor = self.key(tokens).view(batch, length, self.num_heads, -1).transpose(1, 2)
        values: Tensor = self.value(tokens).view(batch, length, self.num_heads, -1).transpose(1, 2)
        series: Tensor = torch.softmax(queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1]), dim=-1)
        attended: Tensor = (self.dropout(series) @ values).transpose(1, 2).reshape(batch, length, -1)
        # The official transformation of the learned scale: sigma in (0, 2], one per step and head.
        sigma: Tensor = torch.pow(3.0, torch.sigmoid(self.sigma(tokens) * 5.0) + 1e-5) - 1.0
        sigma = sigma.transpose(1, 2).unsqueeze(-1)
        gaussian: Tensor = torch.exp(-self.distances.square() / (2.0 * sigma.square())) / (
            math.sqrt(2.0 * math.pi) * sigma
        )
        prior: Tensor = gaussian / gaussian.sum(dim=-1, keepdim=True)
        return self.output(attended), series, prior

    def __call__(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor, Tensor], super().__call__(tokens))
