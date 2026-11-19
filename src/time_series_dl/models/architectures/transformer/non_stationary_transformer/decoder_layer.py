from typing import cast

from torch import Tensor, nn

from ....components.layers.activation import Activation
from ....components.layers.multilayer_perceptron import MultilayerPerceptron
from .de_stationary_attention import DeStationaryAttention
from .de_stationary_factors import DeStationaryFactors


class NonStationaryDecoderLayer(nn.Module):
    """
    Causal de-stationary self-attention rescaled by tau only, de-stationary cross-attention to the encoder rescaled by
    tau and shifted by delta, and a position-wise feed-forward network, each followed by a residual connection and
    layer normalization.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability of the attention weights and sublayer outputs.
    activation : Activation
        Activation of the feed-forward network.
    """

    def __init__(
        self, model_size: int, num_heads: int, feedforward_size: int, dropout: float, activation: Activation
    ) -> None:
        super().__init__()
        self.self_attention: DeStationaryAttention = DeStationaryAttention(
            model_size, num_heads, dropout, is_causal=True
        )
        self.cross_attention: DeStationaryAttention = DeStationaryAttention(
            model_size, num_heads, dropout, is_causal=False
        )
        self.feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (feedforward_size,), model_size, activation, dropout
        )
        self.self_attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.cross_attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.feed_forward_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, memory: Tensor, factors: DeStationaryFactors) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Decoder tokens with shape (batch, length, model_size).
        memory : Tensor
            Encoder output with shape (batch, input_length, model_size).
        factors : DeStationaryFactors
            The de-stationary factors of the windows; delta shifts the scores of the encoder positions.

        Returns
        -------
        Tensor
            Decoder tokens of the shape of ``tokens``.
        """
        attended: Tensor = self.self_attention(tokens, tokens, tokens, factors.scale, None)
        hidden: Tensor = self.self_attention_norm(tokens + self.dropout(attended))
        crossed: Tensor = self.cross_attention(hidden, memory, memory, factors.scale, factors.shift)
        hidden = self.cross_attention_norm(hidden + self.dropout(crossed))
        outputs: Tensor = self.feed_forward_norm(hidden + self.dropout(self.feed_forward(hidden)))
        return outputs

    def __call__(self, tokens: Tensor, memory: Tensor, factors: DeStationaryFactors) -> Tensor:
        return cast(Tensor, super().__call__(tokens, memory, factors))
