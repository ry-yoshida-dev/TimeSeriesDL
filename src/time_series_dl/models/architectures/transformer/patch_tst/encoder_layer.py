from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .token_norm import PatchTSTTokenNorm

if TYPE_CHECKING:
    from .parameters import PatchTSTParameters


class PatchTSTEncoderLayer(nn.Module):
    """
    Post-norm Transformer encoder layer whose normalization may be batch normalization.

    Parameters
    ----------
    parameters : PatchTSTParameters
        The model configuration.
    """

    def __init__(self, parameters: PatchTSTParameters) -> None:
        super().__init__()
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            parameters.model_size, parameters.num_heads, dropout=parameters.dropout, batch_first=True
        )
        self.attention_norm: PatchTSTTokenNorm = PatchTSTTokenNorm(parameters.normalization, parameters.model_size)
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(parameters.model_size, parameters.feedforward_size),
            parameters.activation.build(),
            nn.Dropout(parameters.dropout),
            nn.Linear(parameters.feedforward_size, parameters.model_size),
        )
        self.feedforward_norm: PatchTSTTokenNorm = PatchTSTTokenNorm(parameters.normalization, parameters.model_size)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Apply self-attention and the feed-forward network, each with a residual connection and normalization.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, num_patches, model_size).

        Returns
        -------
        Tensor
            Tokens of the same shape.
        """
        attended: Tensor
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        hidden: Tensor = self.attention_norm(tokens + self.dropout(attended))
        transformed: Tensor = self.feedforward(hidden)
        return self.feedforward_norm(hidden + self.dropout(transformed))

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
