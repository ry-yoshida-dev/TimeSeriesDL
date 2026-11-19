from typing import cast

import torch
from torch import Tensor, nn


class CrossformerSegmentEmbedding(nn.Module):
    """
    Dimension-segment-wise embedding: each channel's series is cut into segments, each segment is mapped linearly to
    a token, a learnable position of every (channel, segment) pair is added, and the tokens are layer-normalized.

    The series is padded at the start with copies of its first value up to a multiple of the segment length, as in
    the official code.

    Parameters
    ----------
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    segment_length : int
        Number of time steps of each segment.
    model_size : int
        Size of each token.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, input_length: int, input_channels: int, segment_length: int, model_size: int) -> None:
        if min(input_length, input_channels, segment_length, model_size) <= 0:
            raise ValueError(
                "input_length, input_channels, segment_length and model_size must be positive, got "
                + f"{input_length}, {input_channels}, {segment_length} and {model_size}"
            )
        super().__init__()
        self.segment_length: int = segment_length
        self.num_segments: int = -(-input_length // segment_length)
        self.padding: int = self.num_segments * segment_length - input_length
        self.projection: nn.Linear = nn.Linear(segment_length, model_size)
        self.position: nn.Parameter = nn.Parameter(torch.randn(input_channels, self.num_segments, model_size))
        self.norm: nn.LayerNorm = nn.LayerNorm(model_size)

    def forward(self, values: Tensor) -> Tensor:
        """
        Embed a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, input_channels).

        Returns
        -------
        Tensor
            Tokens with shape (batch, input_channels, num_segments, model_size).
        """
        padded: Tensor = torch.cat([values[:, :1].expand(-1, self.padding, -1), values], dim=1)
        segments: Tensor = padded.transpose(1, 2).reshape(
            values.shape[0], values.shape[2], self.num_segments, self.segment_length
        )
        tokens: Tensor = self.norm(self.projection(segments) + self.position)
        return tokens

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
