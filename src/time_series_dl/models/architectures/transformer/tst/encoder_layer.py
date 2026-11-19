from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from ....components.layers import MultilayerPerceptron
from ..patch_tst import PatchTSTTokenNorm

if TYPE_CHECKING:
    from .parameters import TSTParameters


class TSTEncoderLayer(nn.Module):
    """
    Post-norm Transformer encoder layer whose normalization is batch normalization by default, as the
    ``TransformerBatchNormEncoderLayer`` of the official code.

    Parameters
    ----------
    parameters : TSTParameters
        The model configuration.
    """

    def __init__(self, parameters: TSTParameters) -> None:
        super().__init__()
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            parameters.model_size, parameters.num_heads, dropout=parameters.dropout, batch_first=True
        )
        self.attention_norm: PatchTSTTokenNorm = PatchTSTTokenNorm(parameters.normalization, parameters.model_size)
        self.feedforward: MultilayerPerceptron = MultilayerPerceptron(
            parameters.model_size,
            (parameters.feedforward_size,),
            parameters.model_size,
            parameters.activation,
            parameters.dropout,
        )
        self.feedforward_norm: PatchTSTTokenNorm = PatchTSTTokenNorm(parameters.normalization, parameters.model_size)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Apply self-attention and the feed-forward network, each with a residual connection and normalization.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Tokens of the same shape.
        """
        attended: Tensor
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        hidden: Tensor = self.attention_norm(tokens + self.dropout(attended))
        return self.feedforward_norm(hidden + self.dropout(self.feedforward(hidden)))

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
