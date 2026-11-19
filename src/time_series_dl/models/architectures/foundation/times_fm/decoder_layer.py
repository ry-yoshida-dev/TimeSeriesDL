from typing import cast

from torch import Tensor, nn

from ....components.t5 import T5LayerNorm
from .attention import TimesFMAttention
from .feed_forward import TimesFMFeedForward


class TimesFMDecoderLayer(nn.Module):
    """
    TimesFM decoder layer: root-mean-square-normalized causal self-attention added to its input, followed by the
    feed-forward network.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    num_heads : int
        Number of attention heads.
    head_size : int
        Size of each head.
    feedforward_size : int
        Hidden size of the feed-forward network.
    layer_norm_epsilon : float
        Epsilon of the root-mean-square normalization.
    """

    def __init__(
        self, model_size: int, num_heads: int, head_size: int, feedforward_size: int, layer_norm_epsilon: float
    ) -> None:
        super().__init__()
        self.attention_norm: T5LayerNorm = T5LayerNorm(model_size, layer_norm_epsilon)
        self.attention: TimesFMAttention = TimesFMAttention(model_size, num_heads, head_size)
        self.feed_forward: TimesFMFeedForward = TimesFMFeedForward(model_size, feedforward_size)

    def forward(self, hidden: Tensor, bias: Tensor, is_padded: Tensor) -> Tensor:
        """
        Transform a batch of patch sequences.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, length, model_size).
        bias : Tensor
            Causal and padding attention bias, broadcastable to (batch, num_heads, length, length).
        is_padded : Tensor
            1.0 at fully padded patches and 0.0 elsewhere, with shape (batch, length).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        hidden = hidden + self.attention(self.attention_norm(hidden), bias)
        return self.feed_forward(hidden, is_padded)

    def __call__(self, hidden: Tensor, bias: Tensor, is_padded: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden, bias, is_padded))
