from typing import cast

import torch
from torch import Tensor, nn


class CrossformerSegmentMerging(nn.Module):
    """
    Merges every ``merge_size`` adjacent segments of each channel into one token, for the next coarser scale.

    When the number of segments is not a multiple of merge_size, the last segments are repeated at the end, as in the
    official code; the merged tokens are concatenated, layer-normalized and mapped linearly to the token size.

    Parameters
    ----------
    num_segments : int
        Number of segments of the input tokens.
    merge_size : int
        Number of segments merged into one.
    model_size : int
        Size of each token.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, num_segments: int, merge_size: int, model_size: int) -> None:
        if min(num_segments, merge_size, model_size) <= 0:
            raise ValueError(
                "num_segments, merge_size and model_size must be positive, got "
                + f"{num_segments}, {merge_size} and {model_size}"
            )
        super().__init__()
        self.merge_size: int = merge_size
        self.output_segments: int = -(-num_segments // merge_size)
        padding: int = self.output_segments * merge_size - num_segments
        indices: Tensor = torch.cat(
            [torch.arange(num_segments), torch.arange(num_segments - padding, num_segments).clamp(min=0)]
        )
        self.indices: Tensor
        self.register_buffer("indices", indices, persistent=False)
        self.norm: nn.LayerNorm = nn.LayerNorm(merge_size * model_size)
        self.projection: nn.Linear = nn.Linear(merge_size * model_size, model_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Merge the segments.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, channels, num_segments, model_size).

        Returns
        -------
        Tensor
            Tokens with shape (batch, channels, output_segments, model_size).
        """
        padded: Tensor = tokens.index_select(2, self.indices)
        merged: Tensor = padded.reshape(tokens.shape[0], tokens.shape[1], self.output_segments, -1)
        outputs: Tensor = self.projection(self.norm(merged))
        return outputs

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
