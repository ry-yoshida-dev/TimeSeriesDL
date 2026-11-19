from typing import cast

from torch import Tensor, nn

from .attention import T5Attention
from .configuration import T5Configuration
from .feed_forward import T5FeedForward
from .layer_norm import T5LayerNorm


class T5EncoderBlock(nn.Module):
    """
    Pre-normalized T5 encoder block: self-attention and a feed-forward network, each added to its input.

    Parameters
    ----------
    configuration : T5Configuration
        The sizes of the stack.
    """

    def __init__(self, configuration: T5Configuration) -> None:
        super().__init__()
        size: int = configuration.model_size
        epsilon: float = configuration.layer_norm_epsilon
        self.self_attention_norm: T5LayerNorm = T5LayerNorm(size, epsilon)
        self.self_attention: T5Attention = T5Attention(
            size, configuration.key_value_size, configuration.num_heads, configuration.dropout
        )
        self.feed_forward_norm: T5LayerNorm = T5LayerNorm(size, epsilon)
        self.feed_forward: T5FeedForward = T5FeedForward(
            size, configuration.feedforward_size, configuration.feed_forward_kind, configuration.dropout
        )
        self.dropout: nn.Dropout = nn.Dropout(configuration.dropout)

    def forward(self, hidden: Tensor, bias: Tensor) -> Tensor:
        """
        Transform a batch of sequences.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, length, model_size).
        bias : Tensor
            Self-attention bias with the mask, broadcastable to (batch, num_heads, length, length).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        normalized: Tensor = self.self_attention_norm(hidden)
        hidden = hidden + self.dropout(self.self_attention(normalized, normalized, bias))
        transformed: Tensor = self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))
        return hidden + transformed

    def __call__(self, hidden: Tensor, bias: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden, bias))
