from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .segment_embedding import CrossformerSegmentEmbedding
from .segment_merging import CrossformerSegmentMerging
from .two_stage_attention_layer import CrossformerTwoStageAttentionLayer

if TYPE_CHECKING:
    from .parameters import CrossformerParameters


class CrossformerEncoder(nn.Module):
    """
    Hierarchical encoder: the segment embedding, then one block per scale, each a two-stage attention layer; every
    block after the first first merges adjacent segments into a coarser scale.

    Parameters
    ----------
    parameters : CrossformerParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: CrossformerParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self.scale_segments: tuple[int, ...] = parameters.count_scale_segments(input_length)
        self.embedding: CrossformerSegmentEmbedding = CrossformerSegmentEmbedding(
            input_length, input_channels, parameters.segment_length, parameters.model_size
        )
        self.merging: nn.ModuleList = nn.ModuleList(
            CrossformerSegmentMerging(num_segments, parameters.merge_size, parameters.model_size)
            for num_segments in self.scale_segments[1:-1]
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            CrossformerTwoStageAttentionLayer(
                num_segments,
                parameters.num_routers,
                parameters.model_size,
                parameters.num_heads,
                parameters.feedforward_size,
                parameters.dropout,
                parameters.activation,
            )
            for num_segments in self.scale_segments[1:]
        )

    def forward(self, values: Tensor) -> list[Tensor]:
        """
        Encode a batch of sequences at every scale.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, input_channels).

        Returns
        -------
        list[Tensor]
            The embedding followed by the output of every block, each with shape
            (batch, input_channels, scale_segments[scale], model_size).
        """
        tokens: Tensor = self.embedding(values)
        scales: list[Tensor] = [tokens]
        for position, layer in enumerate(self.layers):
            if position > 0:
                tokens = self.merging[position - 1](tokens)
            tokens = layer(tokens)
            scales.append(tokens)
        return scales

    def __call__(self, values: Tensor) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(values))
