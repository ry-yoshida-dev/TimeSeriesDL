from typing import cast

from torch import Tensor, nn

from ....components.attention.attention_mechanisms.full import FullAttentionMechanism
from ....components.attention.multi_head_attention import MultiHeadAttention
from ....components.layers.activation import Activation
from ....components.layers.multilayer_perceptron import MultilayerPerceptron
from .two_stage_attention_layer import CrossformerTwoStageAttentionLayer


class CrossformerDecoderLayer(nn.Module):
    """
    Decoder layer at one scale: two-stage self-attention over the output segments, cross-attention of each channel's
    output segments to its encoded segments of that scale, and a feed-forward network, with residual connections and
    layer normalization.

    Parameters
    ----------
    num_segments : int
        Number of output segments.
    num_routers : int
        Number of routers per segment.
    model_size : int
        Size of each token.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward networks of the two-stage attention.
    dropout : float
        Dropout probability of the attention weights and of the sublayer outputs.
    activation : Activation
        Activation of the feed-forward networks.
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
        super().__init__()
        self.self_attention: CrossformerTwoStageAttentionLayer = CrossformerTwoStageAttentionLayer(
            num_segments, num_routers, model_size, num_heads, feedforward_size, dropout, activation
        )
        self.cross_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(False, dropout), model_size, num_heads
        )
        self.feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (model_size,), model_size, activation
        )
        self.cross_attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.feed_forward_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, memory: Tensor) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Output segment tokens with shape (batch, channels, num_segments, model_size).
        memory : Tensor
            Encoded tokens of one scale with shape (batch, channels, encoded_segments, model_size).

        Returns
        -------
        Tensor
            Output segment tokens of the same shape as ``tokens``.
        """
        batch, channels, num_segments, model_size = tokens.shape
        queries: Tensor = self.self_attention(tokens).reshape(batch * channels, num_segments, model_size)
        keys: Tensor = memory.reshape(batch * channels, memory.shape[2], model_size)
        hidden: Tensor = self.cross_attention_norm(queries + self.dropout(self.cross_attention(queries, keys, keys)))
        outputs: Tensor = self.feed_forward_norm(hidden + self.feed_forward(hidden))
        return outputs.reshape(batch, channels, num_segments, model_size)

    def __call__(self, tokens: Tensor, memory: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens, memory))
