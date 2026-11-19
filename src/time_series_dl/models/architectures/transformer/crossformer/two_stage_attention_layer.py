from typing import cast

import torch
from torch import Tensor, nn

from ....components.attention.attention_mechanisms.full import FullAttentionMechanism
from ....components.attention.multi_head_attention import MultiHeadAttention
from ....components.layers.activation import Activation
from ....components.layers.multilayer_perceptron import MultilayerPerceptron


class CrossformerTwoStageAttentionLayer(nn.Module):
    """
    Two-stage attention: self-attention across the segments of each channel, then a router stage across channels.

    In the cross-dimension stage, ``num_routers`` learnable routers per segment gather messages from every channel
    and distribute them back, which costs linear time in the number of channels. Each attention is followed by a
    residual connection, layer normalization, a feed-forward network and another residual connection and layer
    normalization.

    Parameters
    ----------
    num_segments : int
        Number of segments of the input tokens.
    num_routers : int
        Number of routers per segment.
    model_size : int
        Size of each token.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward networks.
    dropout : float
        Dropout probability of the attention weights and of the sublayer outputs.
    activation : Activation
        Activation of the feed-forward networks.

    Raises
    ------
    ValueError
        If num_segments or num_routers is not positive.
    """

    def __init__(
        self,
        num_segments: int,
        num_routers: int,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
        activation: Activation,
    ) -> None:
        if num_segments <= 0 or num_routers <= 0:
            raise ValueError(f"num_segments and num_routers must be positive, got {num_segments} and {num_routers}")
        super().__init__()
        self.time_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(False, dropout), model_size, num_heads
        )
        self.dimension_sender: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(False, dropout), model_size, num_heads
        )
        self.dimension_receiver: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(False, dropout), model_size, num_heads
        )
        self.router: nn.Parameter = nn.Parameter(torch.randn(num_segments, num_routers, model_size))
        self.time_feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (feedforward_size,), model_size, activation
        )
        self.dimension_feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (feedforward_size,), model_size, activation
        )
        self.time_attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.time_feed_forward_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dimension_attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dimension_feed_forward_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Apply both stages.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, channels, num_segments, model_size).

        Returns
        -------
        Tensor
            Tokens of the same shape.
        """
        batch, channels, num_segments, model_size = tokens.shape
        time_tokens: Tensor = tokens.reshape(batch * channels, num_segments, model_size)
        hidden: Tensor = self.time_attention_norm(
            time_tokens + self.dropout(self.time_attention(time_tokens, time_tokens, time_tokens))
        )
        hidden = self.time_feed_forward_norm(hidden + self.dropout(self.time_feed_forward(hidden)))
        dimension_tokens: Tensor = (
            hidden.reshape(batch, channels, num_segments, model_size)
            .transpose(1, 2)
            .reshape(batch * num_segments, channels, model_size)
        )
        routers: Tensor = self.router.repeat(batch, 1, 1)
        buffer: Tensor = self.dimension_sender(routers, dimension_tokens, dimension_tokens)
        received: Tensor = self.dimension_receiver(dimension_tokens, buffer, buffer)
        mixed: Tensor = self.dimension_attention_norm(dimension_tokens + self.dropout(received))
        mixed = self.dimension_feed_forward_norm(mixed + self.dropout(self.dimension_feed_forward(mixed)))
        return mixed.reshape(batch, num_segments, channels, model_size).transpose(1, 2)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
