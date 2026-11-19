from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch.nn.functional as functional
from torch import Tensor, nn

from .decomposable_mixing import TimeMixerDecomposableMixing
from .token_embedding import TimeMixerTokenEmbedding

if TYPE_CHECKING:
    from .parameters import TimeMixerParameters


class TimeMixerEncoder(nn.Module):
    """
    Token embedding of every scale followed by stacked past decomposable mixing blocks.

    Parameters
    ----------
    parameters : TimeMixerParameters
        The model configuration.
    input_length : int
        Number of time steps of the finest scale.
    input_channels : int
        Number of channels embedded into each token.
    is_channel_mixing : bool
        Whether the blocks also mix the token features before the scales are mixed, for tokens of several channels.
    is_finest_output_only : bool
        Whether only the tokens of the finest scale are returned.

    Raises
    ------
    ValueError
        If the coarsest scale would have no time step.
    """

    def __init__(
        self,
        parameters: TimeMixerParameters,
        input_length: int,
        input_channels: int,
        is_channel_mixing: bool,
        is_finest_output_only: bool,
    ) -> None:
        lengths: tuple[int, ...] = parameters.compute_scale_lengths(input_length)
        super().__init__()
        self.lengths: tuple[int, ...] = lengths
        self.window: int = parameters.down_sampling_window
        self.embedding: TimeMixerTokenEmbedding = TimeMixerTokenEmbedding(
            input_channels, parameters.model_size, parameters.dropout
        )
        self.blocks: nn.ModuleList = nn.ModuleList(
            TimeMixerDecomposableMixing(
                lengths,
                parameters.model_size,
                parameters.feedforward_size,
                parameters.kernel_size,
                is_channel_mixing,
                is_finest_output_only and position == parameters.num_layers - 1,
            )
            for position in range(parameters.num_layers)
        )

    def downsample(self, values: Tensor) -> list[Tensor]:
        """
        Build every scale of a batch of sequences by repeated average pooling.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        list[Tensor]
            The scales with shape (batch, length of the scale, channels), from the finest.
        """
        scales: list[Tensor] = [values]
        for _ in self.lengths[1:]:
            pooled: Tensor = functional.avg_pool1d(scales[-1].transpose(1, 2), self.window)
            scales.append(pooled.transpose(1, 2))
        return scales

    def forward(self, scales: list[Tensor]) -> list[Tensor]:
        """
        Encode every scale.

        Parameters
        ----------
        scales : list[Tensor]
            Values with shape (batch, length of the scale, input_channels), from the finest scale.

        Returns
        -------
        list[Tensor]
            Tokens with shape (batch, length of the scale, model_size), from the finest scale.
        """
        tokens: list[Tensor] = [self.embedding(scale) for scale in scales]
        for block in self.blocks:
            tokens = cast(list[Tensor], block(tokens))
        return tokens

    def __call__(self, scales: list[Tensor]) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(scales))
