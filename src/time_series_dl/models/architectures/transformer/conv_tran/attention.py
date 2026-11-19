from typing import cast

import torch
from torch import Tensor, nn

from .relative_position_encoding import RelativePositionEncoding


class ConvTranAttention(nn.Module):
    """
    Multi-head self-attention of ConvTran, with an optional efficient relative position encoding (eRPE) and layer
    normalization of the output instead of an output projection.

    As in the official code, the scores are scaled by ``model_size ** -0.5``, the projections have no bias, and the
    learned bias ``w[i - j]`` of each head is added to the attention weights after the softmax.

    Parameters
    ----------
    relative_encoding : RelativePositionEncoding
        The relative position encoding.
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of heads; it divides model_size.
    length : int
        Number of tokens.
    dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    def __init__(
        self, relative_encoding: RelativePositionEncoding, model_size: int, num_heads: int, length: int, dropout: float
    ) -> None:
        if model_size <= 0 or num_heads <= 0 or model_size % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide model_size, got {num_heads} and {model_size}")
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.num_heads: int = num_heads
        self.scale: float = model_size**-0.5
        self.query_projection: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.key_projection: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.value_projection: nn.Linear = nn.Linear(model_size, model_size, bias=False)
        self.relative_bias_table: nn.Parameter | None = None
        match relative_encoding:
            case RelativePositionEncoding.ERPE:
                self.relative_bias_table = nn.Parameter(torch.zeros(2 * length - 1, num_heads))
            case RelativePositionEncoding.NONE:
                pass
        positions: Tensor = torch.arange(length)
        self.relative_index: Tensor
        self.register_buffer(
            "relative_index", positions.unsqueeze(1) - positions.unsqueeze(0) + length - 1, persistent=False
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.output_norm: nn.LayerNorm = nn.LayerNorm(model_size)

    def compute_relative_bias(self) -> Tensor | None:
        """
        Gather the learned bias of every pair of positions.

        Returns
        -------
        Tensor | None
            The bias with shape (heads, length, length), whose entry (h, i, j) is the bias of head h for the distance
            i - j; None without eRPE.
        """
        if self.relative_bias_table is None:
            return None
        return self.relative_bias_table[self.relative_index].permute(2, 0, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Attend from every token to every token.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Normalized outputs of the same shape.
        """
        queries: Tensor = self._split(self.query_projection(tokens))
        keys: Tensor = self._split(self.key_projection(tokens))
        values: Tensor = self._split(self.value_projection(tokens))
        weights: Tensor = torch.softmax(queries @ keys.transpose(-2, -1) * self.scale, dim=-1)
        relative_bias: Tensor | None = self.compute_relative_bias()
        if relative_bias is not None:
            weights = weights + relative_bias
        attended: Tensor = self.dropout(weights) @ values
        outputs: Tensor = self.output_norm(attended.transpose(1, 2).flatten(start_dim=2))
        return outputs

    def _split(self, tokens: Tensor) -> Tensor:
        return tokens.reshape(tokens.shape[0], tokens.shape[1], self.num_heads, -1).transpose(1, 2)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
