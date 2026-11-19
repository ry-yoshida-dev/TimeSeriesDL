from typing import cast

from torch import Tensor, nn

from .attention_mask import T5AttentionMask
from .configuration import T5Configuration
from .encoder_block import T5EncoderBlock
from .layer_norm import T5LayerNorm
from .relative_position_bias import T5RelativePositionBias


class T5Encoder(nn.Module):
    """
    T5 encoder stack over given input embeddings: bidirectional self-attention with a relative position bias shared
    by every block, and a final layer normalization.

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
            is_bidirectional=True,
        )
        self.blocks: nn.ModuleList = nn.ModuleList(
            T5EncoderBlock(configuration) for _ in range(configuration.num_layers)
        )
        self.final_norm: T5LayerNorm = T5LayerNorm(configuration.model_size, configuration.layer_norm_epsilon)
        self.dropout: nn.Dropout = nn.Dropout(configuration.dropout)

    def forward(self, embeddings: Tensor, is_observed: Tensor) -> Tensor:
        """
        Encode a batch of embedded sequences.

        Parameters
        ----------
        embeddings : Tensor
            Input embeddings with shape (batch, length, model_size).
        is_observed : Tensor
            Boolean mask with shape (batch, length); False positions are never attended to.

        Returns
        -------
        Tensor
            The encoder states with shape (batch, length, model_size).
        """
        length: int = embeddings.shape[1]
        bias: Tensor = self.position_bias(length, length) + T5AttentionMask.from_padding(is_observed, embeddings.dtype)
        hidden: Tensor = self.dropout(embeddings)
        for block in self.blocks:
            hidden = block(hidden, bias)
        outputs: Tensor = self.dropout(self.final_norm(hidden))
        return outputs

    def __call__(self, embeddings: Tensor, is_observed: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(embeddings, is_observed))
