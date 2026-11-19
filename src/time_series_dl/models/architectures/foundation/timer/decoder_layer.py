from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .attention import TimerAttention
from .feed_forward import TimerFeedForward

if TYPE_CHECKING:
    from .parameters import TimerParameters


class TimerDecoderLayer(nn.Module):
    """
    Post-normalized decoder layer of Timer: causal self-attention and a gated feed-forward network, each added to its
    input and followed by a layer normalization.

    Parameters
    ----------
    parameters : TimerParameters
        The model configuration.
    """

    def __init__(self, parameters: TimerParameters) -> None:
        super().__init__()
        size: int = parameters.model_size
        self.attention: TimerAttention = TimerAttention(
            size, parameters.num_heads, parameters.rotary_base, parameters.attention_dropout
        )
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(size, eps=parameters.layer_norm_epsilon)
        self.feed_forward: TimerFeedForward = TimerFeedForward(size, parameters.feedforward_size)
        self.feed_forward_norm: nn.LayerNorm = nn.LayerNorm(size, eps=parameters.layer_norm_epsilon)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Transform a batch of token sequences.

        Parameters
        ----------
        hidden : Tensor
            Hidden states with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        hidden = self.attention_norm(hidden + self.attention(hidden))
        outputs: Tensor = self.feed_forward_norm(hidden + self.feed_forward(hidden))
        return outputs

    def __call__(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden))
