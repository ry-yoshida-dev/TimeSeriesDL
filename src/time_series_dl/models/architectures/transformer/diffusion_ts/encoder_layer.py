from typing import cast

from torch import Tensor, nn

from ....components.attention import FullAttentionMechanism, MultiHeadAttention
from .adaptive_layer_norm import DiffusionTSAdaptiveLayerNorm


class DiffusionTSEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer of Diffusion-TS: self-attention after a normalization modulated by the
    diffusion step, then a feed-forward network after a plain layer normalization.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads; it divides model_size.
    feedforward_size : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability of the attention weights and of the residual branches.
    """

    def __init__(self, model_size: int, num_heads: int, feedforward_size: int, dropout: float) -> None:
        super().__init__()
        self.attention_normalization: DiffusionTSAdaptiveLayerNorm = DiffusionTSAdaptiveLayerNorm(model_size)
        self.attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(is_causal=False, dropout=dropout), model_size, num_heads
        )
        self.feedforward_normalization: nn.LayerNorm = nn.LayerNorm(model_size)
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(model_size, feedforward_size), nn.GELU(), nn.Linear(feedforward_size, model_size)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, step_embedding: Tensor) -> Tensor:
        """
        Encode a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).
        step_embedding : Tensor
            Embedding of each sample's diffusion step with shape (batch, model_size).

        Returns
        -------
        Tensor
            The encoded tokens with the shape of the input.
        """
        normalized: Tensor = self.attention_normalization(tokens, step_embedding)
        tokens = tokens + self.dropout(self.attention(normalized, normalized, normalized))
        encoded: Tensor = tokens + self.dropout(self.feedforward(self.feedforward_normalization(tokens)))
        return encoded

    def __call__(self, tokens: Tensor, step_embedding: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens, step_embedding))
