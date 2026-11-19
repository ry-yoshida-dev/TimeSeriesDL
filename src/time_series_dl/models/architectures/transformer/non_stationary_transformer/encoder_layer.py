from typing import cast

from torch import Tensor, nn

from ....components.layers.activation import Activation
from ....components.layers.multilayer_perceptron import MultilayerPerceptron
from .de_stationary_attention import DeStationaryAttention
from .de_stationary_factors import DeStationaryFactors


class NonStationaryEncoderLayer(nn.Module):
    """
    De-stationary self-attention and a position-wise feed-forward network, each followed by a residual connection
    and layer normalization.

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
        self.attention: DeStationaryAttention = DeStationaryAttention(model_size, num_heads, dropout, is_causal=False)
        self.feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (feedforward_size,), model_size, activation, dropout
        )
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.feed_forward_norm: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, factors: DeStationaryFactors) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, input_length, model_size).
        factors : DeStationaryFactors
            The de-stationary factors of the windows.

        Returns
        -------
        Tensor
            Tokens of the same shape.
        """
        attended: Tensor = self.attention(tokens, tokens, tokens, factors.scale, factors.shift)
        hidden: Tensor = self.attention_norm(tokens + self.dropout(attended))
        outputs: Tensor = self.feed_forward_norm(hidden + self.dropout(self.feed_forward(hidden)))
        return outputs

    def __call__(self, tokens: Tensor, factors: DeStationaryFactors) -> Tensor:
        return cast(Tensor, super().__call__(tokens, factors))
