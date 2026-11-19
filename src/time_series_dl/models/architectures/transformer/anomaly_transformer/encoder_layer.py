from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .anomaly_attention import AnomalyAttention

if TYPE_CHECKING:
    from .parameters import AnomalyTransformerParameters


class AnomalyTransformerEncoderLayer(nn.Module):
    """
    Anomaly-attention followed by a position-wise feed-forward network, each with a residual connection and
    post-layer normalization.

    Parameters
    ----------
    parameters : AnomalyTransformerParameters
        The model configuration.
    length : int
        Number of steps of the windows.
    """

    def __init__(self, parameters: AnomalyTransformerParameters, length: int) -> None:
        super().__init__()
        self.attention: AnomalyAttention = AnomalyAttention(
            parameters.model_size, parameters.num_heads, length, parameters.dropout
        )
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(parameters.model_size, parameters.feedforward_size),
            parameters.activation.build(),
            nn.Dropout(parameters.dropout),
            nn.Linear(parameters.feedforward_size, parameters.model_size),
            nn.Dropout(parameters.dropout),
        )
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.feedforward_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            The encoded tokens, and the series and prior associations of the layer.
        """
        attended: Tensor
        series: Tensor
        prior: Tensor
        attended, series, prior = self.attention(tokens)
        normalized: Tensor = self.attention_norm(tokens + self.dropout(attended))
        encoded: Tensor = self.feedforward_norm(normalized + self.feedforward(normalized))
        return encoded, series, prior

    def __call__(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor, Tensor], super().__call__(tokens))
