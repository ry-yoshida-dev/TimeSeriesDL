from typing import cast

from torch import Tensor, nn

from .attention import T5Attention
from .configuration import T5Configuration
from .decoder_layer_cache import T5DecoderLayerCache
from .feed_forward import T5FeedForward
from .layer_norm import T5LayerNorm


class T5DecoderBlock(nn.Module):
    """
    Pre-normalized T5 decoder block: causal self-attention, attention to the encoder states and a feed-forward
    network, each added to its input.

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
        self.cross_attention_norm: T5LayerNorm = T5LayerNorm(size, epsilon)
        self.cross_attention: T5Attention = T5Attention(
            size, configuration.key_value_size, configuration.num_heads, configuration.dropout
        )
        self.feed_forward_norm: T5LayerNorm = T5LayerNorm(size, epsilon)
        self.feed_forward: T5FeedForward = T5FeedForward(
            size, configuration.feedforward_size, configuration.feed_forward_kind, configuration.dropout
        )
        self.dropout: nn.Dropout = nn.Dropout(configuration.dropout)

    def forward(self, hidden: Tensor, self_bias: Tensor, encoder_states: Tensor, cross_bias: Tensor) -> Tensor:
        """
        Transform a batch of decoder sequences.

        Parameters
        ----------
        hidden : Tensor
            Decoder hidden states with shape (batch, length, model_size).
        self_bias : Tensor
            Causal self-attention bias, broadcastable to (batch, num_heads, length, length).
        encoder_states : Tensor
            Encoder outputs with shape (batch, encoder_length, model_size).
        cross_bias : Tensor
            Mask of the encoder positions, broadcastable to (batch, num_heads, length, encoder_length).

        Returns
        -------
        Tensor
            The new hidden states.
        """
        normalized: Tensor = self.self_attention_norm(hidden)
        hidden = hidden + self.dropout(self.self_attention(normalized, normalized, self_bias))
        attended: Tensor = self.cross_attention(self.cross_attention_norm(hidden), encoder_states, cross_bias)
        hidden = hidden + self.dropout(attended)
        transformed: Tensor = self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))
        return hidden + transformed

    def decode_step(self, hidden: Tensor, self_bias: Tensor, cross_bias: Tensor, cache: T5DecoderLayerCache) -> Tensor:
        """
        Transform the newest decoder positions, attending to the cached earlier positions; equal to the last positions
        of ``forward`` over the whole sequence.

        Parameters
        ----------
        hidden : Tensor
            Hidden states of the new positions with shape (batch, new_length, model_size).
        self_bias : Tensor
            Self-attention bias of the new positions to every decoded position, broadcastable to
            (batch, num_heads, new_length, length).
        cross_bias : Tensor
            Mask of the encoder positions, broadcastable to (batch, num_heads, new_length, encoder_length).
        cache : T5DecoderLayerCache
            Keys and values of the earlier positions and of the encoder states; the new positions are appended.

        Returns
        -------
        Tensor
            The new hidden states of the new positions.
        """
        normalized: Tensor = self.self_attention_norm(hidden)
        key, value = cache.append(*self.self_attention.project_keys_values(normalized))
        hidden = hidden + self.dropout(self.self_attention.attend(normalized, key, value, self_bias))
        attended: Tensor = self.cross_attention.attend(
            self.cross_attention_norm(hidden), cache.cross_key, cache.cross_value, cross_bias
        )
        hidden = hidden + self.dropout(attended)
        transformed: Tensor = self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))
        return hidden + transformed

    def __call__(self, hidden: Tensor, self_bias: Tensor, encoder_states: Tensor, cross_bias: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden, self_bias, encoder_states, cross_bias))
