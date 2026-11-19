from typing import cast

from torch import Tensor, nn

from ....components.attention.attention_mechanism import AttentionMechanism
from ....components.attention.multi_head_attention import MultiHeadAttention
from ....components.layers.activation import Activation
from ....components.layers.multilayer_perceptron import MultilayerPerceptron
from ....components.layers.series_decomposition import SeriesDecomposition


class DecompositionEncoderLayer(nn.Module):
    """
    Self-attention and a feed-forward network, each followed by a residual connection and a series decomposition
    that discards the trend.

    Parameters
    ----------
    mechanism : AttentionMechanism
        The self-attention within each head.
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward network.
    moving_average_size : int
        Length of the moving average of the decompositions.
    dropout : float
        Dropout probability of the sublayer outputs.
    activation : Activation
        Activation of the feed-forward network.
    """

    def __init__(
        self,
        mechanism: AttentionMechanism,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        moving_average_size: int,
        dropout: float,
        activation: Activation,
    ) -> None:
        super().__init__()
        self.attention: MultiHeadAttention = MultiHeadAttention(mechanism, model_size, num_heads)
        self.feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (feedforward_size,), model_size, activation, dropout
        )
        self.decomposition: SeriesDecomposition = SeriesDecomposition(moving_average_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            The seasonal part of the output tokens, of the same shape.
        """
        seasonal: Tensor = self.decomposition(tokens + self.dropout(self.attention(tokens, tokens, tokens)))[0]
        outputs: Tensor = self.decomposition(seasonal + self.dropout(self.feed_forward(seasonal)))[0]
        return outputs

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
