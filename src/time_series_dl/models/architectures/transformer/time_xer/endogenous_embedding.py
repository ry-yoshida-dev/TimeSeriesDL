from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.layers import SinusoidalPositionalEncoding

if TYPE_CHECKING:
    from .parameters import TimeXerParameters


class TimeXerEndogenousEmbedding(nn.Module):
    """
    Endogenous tokens of TimeXer: each endogenous series is cut into non-overlapping patches, embedded linearly with
    the sinusoidal position encoding, and followed by a learnable global token of its variate.

    A series whose length is not a multiple of the patch length is padded at the start with copies of its first value,
    so the most recent steps fill the last patch.

    Parameters
    ----------
    parameters : TimeXerParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    num_variates : int
        Number of endogenous variates, each with its own global token.

    Raises
    ------
    ValueError
        If input_length or num_variates is not positive.
    """

    def __init__(self, parameters: TimeXerParameters, input_length: int, num_variates: int) -> None:
        if input_length <= 0 or num_variates <= 0:
            raise ValueError(f"input_length and num_variates must be positive, got {input_length} and {num_variates}")
        super().__init__()
        self.patch_length: int = parameters.patch_length
        self.padding: int = -input_length % parameters.patch_length
        self.num_patches: int = (input_length + self.padding) // parameters.patch_length
        self.model_size: int = parameters.model_size
        self.embedding: nn.Linear = nn.Linear(parameters.patch_length, parameters.model_size, bias=False)
        self.position: SinusoidalPositionalEncoding = SinusoidalPositionalEncoding(
            parameters.model_size, self.num_patches
        )
        self.global_tokens: nn.Parameter = nn.Parameter(torch.randn(num_variates, 1, parameters.model_size))
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    def forward(self, series: Tensor) -> Tensor:
        """
        Embed the endogenous series of a batch.

        Parameters
        ----------
        series : Tensor
            Endogenous series with shape (batch, num_variates, input_length).

        Returns
        -------
        Tensor
            Patch tokens followed by the global token of each variate, with shape
            (batch * num_variates, num_patches + 1, model_size).
        """
        batch: int = series.shape[0]
        num_variates: int = series.shape[1]
        padded: Tensor = torch.cat([series[..., :1].expand(-1, -1, self.padding), series], dim=-1)
        patches: Tensor = padded.unfold(-1, self.patch_length, self.patch_length)
        tokens: Tensor = self.position(
            self.embedding(patches).reshape(batch * num_variates, self.num_patches, self.model_size)
        )
        global_tokens: Tensor = self.global_tokens.expand(batch, -1, -1, -1).reshape(-1, 1, self.model_size)
        embedded: Tensor = self.dropout(torch.cat([tokens, global_tokens], dim=1))
        return embedded

    def __call__(self, series: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(series))
