from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.attention import FullAttentionMechanism, MultiHeadAttention
from ....components.layers import MultilayerPerceptron

if TYPE_CHECKING:
    from .parameters import TimeXerParameters


class TimeXerEncoderLayer(nn.Module):
    """
    TimeXer encoder layer: self-attention among the patch tokens and the global token of each endogenous variate,
    cross-attention from each global token to the exogenous variate tokens, and a feed-forward network, each with a
    residual connection and layer normalization.

    Parameters
    ----------
    parameters : TimeXerParameters
        The model configuration.
    """

    def __init__(self, parameters: TimeXerParameters) -> None:
        super().__init__()
        self.self_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(False, parameters.dropout), parameters.model_size, parameters.num_heads
        )
        self.cross_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(False, parameters.dropout), parameters.model_size, parameters.num_heads
        )
        self.feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            parameters.model_size,
            (parameters.feedforward_size,),
            parameters.model_size,
            parameters.activation,
            parameters.dropout,
        )
        self.self_attention_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.cross_attention_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.feed_forward_norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    def forward(self, tokens: Tensor, exogenous: Tensor) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Patch tokens followed by the global token of each endogenous variate, with shape
            (batch * num_endogenous, num_patches + 1, model_size), grouped by batch first.
        exogenous : Tensor
            Exogenous variate tokens with shape (batch, num_exogenous, model_size).

        Returns
        -------
        Tensor
            Tokens with the shape of ``tokens``.
        """
        batch: int = exogenous.shape[0]
        model_size: int = exogenous.shape[2]
        attended: Tensor = self.self_attention_norm(tokens + self.dropout(self.self_attention(tokens, tokens, tokens)))
        global_tokens: Tensor = attended[:, -1:]
        queries: Tensor = global_tokens.reshape(batch, -1, model_size)
        crossed: Tensor = self.dropout(self.cross_attention(queries, exogenous, exogenous)).reshape(-1, 1, model_size)
        updated_global: Tensor = self.cross_attention_norm(global_tokens + crossed)
        hidden: Tensor = torch.cat([attended[:, :-1], updated_global], dim=1)
        outputs: Tensor = self.feed_forward_norm(hidden + self.dropout(self.feed_forward(hidden)))
        return outputs

    def __call__(self, tokens: Tensor, exogenous: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens, exogenous))
