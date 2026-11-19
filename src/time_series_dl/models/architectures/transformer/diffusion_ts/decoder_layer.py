from typing import cast

from torch import Tensor, nn

from ....components.attention import FullAttentionMechanism, MultiHeadAttention
from ..etsformer.frequency_attention import ETSformerFrequencyAttention
from .adaptive_layer_norm import DiffusionTSAdaptiveLayerNorm
from .decoder_layer_output import DiffusionTSDecoderLayerOutput
from .trend_block import DiffusionTSTrendBlock


class DiffusionTSDecoderLayer(nn.Module):
    """
    Decoder layer of Diffusion-TS: self-attention and cross-attention to the encoder, each after a normalization
    modulated by the diffusion step; a pointwise convolution over the time steps then splits the tokens into a trend
    input, turned into a polynomial trend, and a seasonal input, turned into its largest Fourier components; finally a
    feed-forward network, after which the mean over time is removed and returned as the layer's level.

    Parameters
    ----------
    length : int
        Number of time steps.
    output_channels : int
        Number of channels of the trend and level.
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads; it divides model_size.
    feedforward_size : int
        Hidden size of the feed-forward network.
    trend_degree : int
        Highest power of the polynomial trend.
    num_frequencies : int
        Number of Fourier components of the seasonality.
    dropout : float
        Dropout probability of the attention weights and of the residual branches.
    """

    def __init__(
        self,
        length: int,
        output_channels: int,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        trend_degree: int,
        num_frequencies: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_normalization: DiffusionTSAdaptiveLayerNorm = DiffusionTSAdaptiveLayerNorm(model_size)
        self.self_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(is_causal=False, dropout=dropout), model_size, num_heads
        )
        self.cross_attention_normalization: DiffusionTSAdaptiveLayerNorm = DiffusionTSAdaptiveLayerNorm(model_size)
        self.cross_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(is_causal=False, dropout=dropout), model_size, num_heads
        )
        self.split: nn.Conv1d = nn.Conv1d(length, 2 * length, kernel_size=1)
        self.trend: DiffusionTSTrendBlock = DiffusionTSTrendBlock(length, model_size, output_channels, trend_degree)
        self.seasonal: ETSformerFrequencyAttention = ETSformerFrequencyAttention(top_k=num_frequencies, horizon=0)
        self.feedforward_normalization: nn.LayerNorm = nn.LayerNorm(model_size)
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(model_size, feedforward_size), nn.GELU(), nn.Linear(feedforward_size, model_size)
        )
        self.level: nn.Linear = nn.Linear(model_size, output_channels)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, encoded: Tensor, step_embedding: Tensor) -> DiffusionTSDecoderLayerOutput:
        """
        Decode a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).
        encoded : Tensor
            Encoder outputs with shape (batch, length, model_size).
        step_embedding : Tensor
            Embedding of each sample's diffusion step with shape (batch, model_size).

        Returns
        -------
        DiffusionTSDecoderLayerOutput
            The tokens without their mean, the level, the trend and the seasonality.
        """
        normalized: Tensor = self.self_attention_normalization(tokens, step_embedding)
        tokens = tokens + self.dropout(self.self_attention(normalized, normalized, normalized))
        normalized = self.cross_attention_normalization(tokens, step_embedding)
        tokens = tokens + self.dropout(self.cross_attention(normalized, encoded, encoded))
        trend_input: Tensor
        seasonal_input: Tensor
        trend_input, seasonal_input = self.split(tokens).chunk(2, dim=1)
        trend: Tensor = self.trend(trend_input)
        seasonal: Tensor = self.seasonal(seasonal_input)
        tokens = tokens + self.dropout(self.feedforward(self.feedforward_normalization(tokens)))
        mean: Tensor = tokens.mean(dim=1, keepdim=True)
        return DiffusionTSDecoderLayerOutput(
            tokens=tokens - mean, mean=self.level(mean), trend=trend, seasonal=seasonal
        )

    def __call__(self, tokens: Tensor, encoded: Tensor, step_embedding: Tensor) -> DiffusionTSDecoderLayerOutput:
        return cast(DiffusionTSDecoderLayerOutput, super().__call__(tokens, encoded, step_embedding))
