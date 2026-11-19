from typing import cast

from torch import Tensor, nn

from .attention_mask import T5AttentionMask
from .configuration import T5Configuration
from .decoder_block import T5DecoderBlock
from .decoder_layer_cache import T5DecoderLayerCache
from .layer_norm import T5LayerNorm
from .relative_position_bias import T5RelativePositionBias


class T5Decoder(nn.Module):
    """
    T5 decoder stack over given input embeddings: causal self-attention with a unidirectional relative position bias
    shared by every block, attention to the encoder states, and a final layer normalization.

    Parameters
    ----------
    configuration : T5Configuration
        The sizes of the stack.
    """

    def __init__(self, configuration: T5Configuration) -> None:
        super().__init__()
        self.position_bias: T5RelativePositionBias = T5RelativePositionBias(
            configuration.num_heads,
            configuration.relative_attention_num_buckets,
            configuration.relative_attention_max_distance,
            is_bidirectional=False,
        )
        self.blocks: nn.ModuleList = nn.ModuleList(
            T5DecoderBlock(configuration) for _ in range(configuration.num_layers)
        )
        self.final_norm: T5LayerNorm = T5LayerNorm(configuration.model_size, configuration.layer_norm_epsilon)
        self.dropout: nn.Dropout = nn.Dropout(configuration.dropout)

    def forward(self, embeddings: Tensor, encoder_states: Tensor, is_encoder_observed: Tensor) -> Tensor:
        """
        Decode a batch of embedded sequences.

        Parameters
        ----------
        embeddings : Tensor
            Decoder input embeddings with shape (batch, length, model_size).
        encoder_states : Tensor
            Encoder outputs with shape (batch, encoder_length, model_size).
        is_encoder_observed : Tensor
            Boolean mask of the encoder positions with shape (batch, encoder_length).

        Returns
        -------
        Tensor
            The decoder states with shape (batch, length, model_size).
        """
        length: int = embeddings.shape[1]
        self_bias: Tensor = self.position_bias(length, length) + T5AttentionMask.causal(
            length, embeddings.dtype, embeddings.device
        )
        cross_bias: Tensor = T5AttentionMask.from_padding(is_encoder_observed, embeddings.dtype)
        hidden: Tensor = self.dropout(embeddings)
        for block in self.blocks:
            hidden = block(hidden, self_bias, encoder_states, cross_bias)
        outputs: Tensor = self.dropout(self.final_norm(hidden))
        return outputs

    def create_cache(self, encoder_states: Tensor) -> tuple[T5DecoderLayerCache, ...]:
        """
        Start incremental decoding: project the encoder states of every block once.

        Parameters
        ----------
        encoder_states : Tensor
            Encoder outputs with shape (batch, encoder_length, model_size).

        Returns
        -------
        tuple[T5DecoderLayerCache, ...]
            One empty cache per block, which ``decode_step`` fills.
        """
        caches: list[T5DecoderLayerCache] = []
        for block in self.blocks:
            assert isinstance(block, T5DecoderBlock)
            cross_key, cross_value = block.cross_attention.project_keys_values(encoder_states)
            empty: Tensor = cross_key[:, :, :0]
            caches.append(
                T5DecoderLayerCache(self_key=empty, self_value=empty, cross_key=cross_key, cross_value=cross_value)
            )
        return tuple(caches)

    def decode_step(
        self, embeddings: Tensor, is_encoder_observed: Tensor, caches: tuple[T5DecoderLayerCache, ...]
    ) -> Tensor:
        """
        Decode the next positions given the earlier ones in the caches; equal to the last positions of ``forward``
        over the whole sequence.

        Parameters
        ----------
        embeddings : Tensor
            Input embeddings of the new positions with shape (batch, new_length, model_size).
        is_encoder_observed : Tensor
            Boolean mask of the encoder positions with shape (batch, encoder_length).
        caches : tuple[T5DecoderLayerCache, ...]
            The caches of ``create_cache``, holding the earlier positions; the new positions are appended.

        Returns
        -------
        Tensor
            The decoder states of the new positions with shape (batch, new_length, model_size).
        """
        new_length: int = embeddings.shape[1]
        length: int = caches[0].self_key.shape[2] + new_length
        self_bias: Tensor = (
            self.position_bias(length, length) + T5AttentionMask.causal(length, embeddings.dtype, embeddings.device)
        )[:, :, -new_length:]
        cross_bias: Tensor = T5AttentionMask.from_padding(is_encoder_observed, embeddings.dtype)
        hidden: Tensor = self.dropout(embeddings)
        for block, cache in zip(self.blocks, caches, strict=True):
            assert isinstance(block, T5DecoderBlock)
            hidden = block.decode_step(hidden, self_bias, cross_bias, cache)
        outputs: Tensor = self.dropout(self.final_norm(hidden))
        return outputs

    def __call__(self, embeddings: Tensor, encoder_states: Tensor, is_encoder_observed: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(embeddings, encoder_states, is_encoder_observed))
