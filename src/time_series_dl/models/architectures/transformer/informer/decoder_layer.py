from typing import cast

from torch import Tensor, nn

from ....components.attention.attention_mechanism import AttentionMechanism
from ....components.attention.multi_head_attention import MultiHeadAttention
from ....components.layers.activation import Activation
from ....components.layers.multilayer_perceptron import MultilayerPerceptron


class InformerDecoderLayer(nn.Module):
    """
    Masked self-attention, cross-attention to the encoder output and a position-wise feed-forward network, each
    followed by a residual connection and layer normalization.

    Parameters
    ----------
    self_mechanism : AttentionMechanism
        The causal self-attention within each head.
    cross_mechanism : AttentionMechanism
        The cross-attention within each head.
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability of the sublayer outputs.
    activation : Activation
        Activation of the feed-forward network.
    """

    def __init__(
        self,
        self_mechanism: AttentionMechanism,
        cross_mechanism: AttentionMechanism,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
        activation: Activation,
    ) -> None:
        super().__init__()
        self.self_attention: MultiHeadAttention = MultiHeadAttention(self_mechanism, model_size, num_heads)
        self.cross_attention: MultiHeadAttention = MultiHeadAttention(cross_mechanism, model_size, num_heads)
        self.feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (feedforward_size,), model_size, activation, dropout
        )
        self.self_attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.cross_attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.feed_forward_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, memory: Tensor) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Decoder tokens with shape (batch, length, model_size).
        memory : Tensor
            Encoder output with shape (batch, memory_length, model_size).

        Returns
        -------
        Tensor
            Decoder tokens of the shape of ``tokens``.
        """
        hidden: Tensor = self.self_attention_norm(tokens + self.dropout(self.self_attention(tokens, tokens, tokens)))
        hidden = self.cross_attention_norm(hidden + self.dropout(self.cross_attention(hidden, memory, memory)))
        outputs: Tensor = self.feed_forward_norm(hidden + self.dropout(self.feed_forward(hidden)))
        return outputs

    def __call__(self, tokens: Tensor, memory: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens, memory))
