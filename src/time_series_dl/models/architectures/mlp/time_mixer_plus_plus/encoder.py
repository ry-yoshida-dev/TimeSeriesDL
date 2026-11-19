from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from ....components.attention import FullAttentionMechanism, MultiHeadAttention
from .mixer_block import TimeMixerPlusPlusMixerBlock

if TYPE_CHECKING:
    from .parameters import TimeMixerPlusPlusParameters


class TimeMixerPlusPlusEncoder(nn.Module):
    """
    Input projection and mixer blocks of TimeMixer++.

    The input is down-sampled by a strided convolution, repeatedly, into coarser scales; the coarsest scale is mixed
    across channels by self-attention among the variates (each variate's series is one token, with a residual
    connection); every scale is embedded into tokens by a circular convolution over channels; and the mixer blocks
    mix the tokens of all scales.

    Parameters
    ----------
    parameters : TimeMixerPlusPlusParameters
        The model configuration.
    length : int
        Number of time steps of the input.
    channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If the input is too short for the down-samplings.
    """

    def __init__(self, parameters: TimeMixerPlusPlusParameters, length: int, channels: int) -> None:
        lengths: tuple[int, ...] = parameters.scale_lengths(length)
        super().__init__()
        self.lengths: tuple[int, ...] = lengths
        self.down_sampling: nn.Conv1d = nn.Conv1d(
            channels, channels, kernel_size=3, stride=2, padding=1, padding_mode="circular", bias=False
        )
        self.channel_attention: MultiHeadAttention = MultiHeadAttention(
            FullAttentionMechanism(is_causal=False, dropout=parameters.dropout), lengths[-1], 1
        )
        self.embedding: nn.Conv1d = nn.Conv1d(
            channels, parameters.model_size, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.blocks: nn.ModuleList = nn.ModuleList(
            TimeMixerPlusPlusMixerBlock(parameters, lengths) for _ in range(parameters.num_layers)
        )

    def forward(self, values: Tensor) -> list[Tensor]:
        """
        Encode a batch of sequences at every scale.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        list[Tensor]
            Tokens from the finest to the coarsest scale, each with shape (batch, length of the scale, model_size).
        """
        series: list[Tensor] = [values.transpose(1, 2)]
        for _ in range(len(self.lengths) - 1):
            series.append(self.down_sampling(series[-1]))
        coarsest: Tensor = series[-1]
        series[-1] = coarsest + self.channel_attention(coarsest, coarsest, coarsest)
        tokens: list[Tensor] = [self.dropout(self.embedding(sequence)).transpose(1, 2) for sequence in series]
        for block in self.blocks:
            tokens = cast(list[Tensor], block(tokens))
        return tokens

    def __call__(self, values: Tensor) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(values))
