from typing import cast

from torch import Tensor, nn

from ....components.attention.attention_mechanism import AttentionMechanism
from ....components.attention.multi_head_attention import MultiHeadAttention
from ....components.layers.activation import Activation
from ....components.layers.multilayer_perceptron import MultilayerPerceptron
from ....components.layers.series_decomposition import SeriesDecomposition


class DecompositionDecoderLayer(nn.Module):
    """
    Self-attention, cross-attention to the encoder output and a feed-forward network, each followed by a residual
    connection and a series decomposition; the three trends are summed and projected to the output channels.

    Parameters
    ----------
    self_mechanism : AttentionMechanism
        The self-attention within each head.
    cross_mechanism : AttentionMechanism
        The cross-attention within each head.
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
    output_channels : int
        Number of channels of the projected trend.
    """

    def __init__(
        self,
        self_mechanism: AttentionMechanism,
        cross_mechanism: AttentionMechanism,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        moving_average_size: int,
        dropout: float,
        activation: Activation,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.self_attention: MultiHeadAttention = MultiHeadAttention(self_mechanism, model_size, num_heads)
        self.cross_attention: MultiHeadAttention = MultiHeadAttention(cross_mechanism, model_size, num_heads)
        self.feed_forward: MultilayerPerceptron = MultilayerPerceptron(
            model_size, (feedforward_size,), model_size, activation, dropout
        )
        self.decomposition: SeriesDecomposition = SeriesDecomposition(moving_average_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.trend_projection: nn.Conv1d = nn.Conv1d(
            model_size, output_channels, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )

    def forward(self, tokens: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the layer.

        Parameters
        ----------
        tokens : Tensor
            Seasonal decoder tokens with shape (batch, length, model_size).
        memory : Tensor
            Encoder output with shape (batch, memory_length, model_size).

        Returns
        -------
        tuple[Tensor, Tensor]
            The seasonal tokens of the shape of ``tokens``, and the trend increment with shape
            (batch, length, output_channels).
        """
        seasonal: Tensor
        first_trend: Tensor
        second_trend: Tensor
        third_trend: Tensor
        seasonal, first_trend = self.decomposition(tokens + self.dropout(self.self_attention(tokens, tokens, tokens)))
        seasonal, second_trend = self.decomposition(
            seasonal + self.dropout(self.cross_attention(seasonal, memory, memory))
        )
        seasonal, third_trend = self.decomposition(seasonal + self.dropout(self.feed_forward(seasonal)))
        trend: Tensor = first_trend + second_trend + third_trend
        projected: Tensor = self.trend_projection(trend.transpose(1, 2))
        return seasonal, projected.transpose(1, 2)

    def __call__(self, tokens: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(tokens, memory))
