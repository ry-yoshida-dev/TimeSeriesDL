from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .decoder_layer import TimesFMDecoderLayer
from .residual_block import TimesFMResidualBlock

if TYPE_CHECKING:
    from .parameters import TimesFMParameters

_MINIMUM_SCALE: Final[float] = 1e-6
_MINIMUM_OBSERVED_PER_PATCH: Final[int] = 3


class TimesFMModel(nn.Module):
    """
    TimesFM (Das et al., 2024): a decoder-only transformer over patches. The series is normalized by the mean and
    standard deviation of its first patch with at least three observed values, each patch is embedded with its
    padding mask by a residual block and the frequency embedding is added; causal transformer layers follow, and the
    output of the last patch is projected to the mean and the quantiles of the next ``output_patch_length`` values.

    Parameters
    ----------
    parameters : TimesFMParameters
        The model configuration.
    """

    def __init__(self, parameters: TimesFMParameters) -> None:
        super().__init__()
        self.patch_length: int = parameters.patch_length
        self.output_patch_length: int = parameters.output_patch_length
        self.num_outputs: int = 1 + len(parameters.quantile_levels)
        self.frequency: int = parameters.frequency.index
        self.input_embedding: TimesFMResidualBlock = TimesFMResidualBlock(
            2 * parameters.patch_length, parameters.feedforward_size, parameters.model_size
        )
        self.frequency_embedding: nn.Embedding = nn.Embedding(parameters.num_frequencies, parameters.model_size)
        self.layers: nn.ModuleList = nn.ModuleList(
            TimesFMDecoderLayer(
                parameters.model_size,
                parameters.num_heads,
                parameters.head_size,
                parameters.feedforward_size,
                parameters.layer_norm_epsilon,
            )
            for _ in range(parameters.num_layers)
        )
        self.output_projection: TimesFMResidualBlock = TimesFMResidualBlock(
            parameters.model_size, parameters.feedforward_size, parameters.output_patch_length * self.num_outputs
        )

    def forward(self, context: Tensor, is_padded: Tensor) -> Tensor:
        """
        Forecast the values after the end of each context.

        Parameters
        ----------
        context : Tensor
            Past values with shape (batch, length).
        is_padded : Tensor
            Boolean mask with shape (batch, length); True marks values that are missing or padding.

        Returns
        -------
        Tensor
            The mean followed by the quantiles of the next ``output_patch_length`` values, with shape
            (batch, output_patch_length, 1 + number of quantile levels), in the scale of the context.
        """
        remainder: int = context.shape[1] % self.patch_length
        padding: Tensor = is_padded.to(context.dtype)
        if remainder != 0:
            # Contexts are left-padded to whole patches, which is what padding to the context length does.
            context = functional.pad(context, (self.patch_length - remainder, 0), value=0.0)
            padding = functional.pad(padding, (self.patch_length - remainder, 0), value=1.0)
        batch: int = context.shape[0]
        patches: Tensor = context.reshape(batch, -1, self.patch_length)
        patch_padding: Tensor = padding.reshape(batch, -1, self.patch_length)
        patches = torch.where(patch_padding == 1.0, 0.0, patches)
        location, scale = self._compute_statistics(patches, patch_padding)
        normalized: Tensor = (patches - location[:, None, None]) / scale[:, None, None] * (1.0 - patch_padding)
        hidden: Tensor = self.input_embedding(torch.cat([normalized, patch_padding], dim=-1))
        hidden = hidden + self.frequency_embedding.weight[self.frequency]
        # A patch counts as padded only if all of its values are padded.
        is_patch_padded: Tensor = patch_padding.min(dim=-1).values
        bias: Tensor = self._attention_bias(is_patch_padded)
        for layer in self.layers:
            hidden = layer(hidden, bias, is_patch_padded)
        outputs: Tensor = self.output_projection(hidden[:, -1]).reshape(
            batch, self.output_patch_length, self.num_outputs
        )
        return outputs * scale[:, None, None] + location[:, None, None]

    def __call__(self, context: Tensor, is_padded: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(context, is_padded))

    @staticmethod
    def _compute_statistics(patches: Tensor, patch_padding: Tensor) -> tuple[Tensor, Tensor]:
        observed: Tensor = 1.0 - patch_padding
        is_enough: Tensor = observed.sum(dim=-1) >= _MINIMUM_OBSERVED_PER_PATCH
        # The first patch with at least three observed values, or the last patch if there is none.
        first: Tensor = torch.where(is_enough.any(dim=1), is_enough.to(torch.int32).argmax(dim=1), patches.shape[1] - 1)
        rows: Tensor = torch.arange(patches.shape[0], device=patches.device)
        values: Tensor = patches[rows, first]
        mask: Tensor = observed[rows, first]
        count: Tensor = mask.sum(dim=-1).clamp(min=1.0)
        location: Tensor = (values * mask).sum(dim=-1) / count
        variance: Tensor = (((values - location.unsqueeze(-1)) * mask) ** 2).sum(dim=-1) / count
        return location, variance.clamp(min=0.0).sqrt().clamp(min=_MINIMUM_SCALE)

    @staticmethod
    def _attention_bias(is_patch_padded: Tensor) -> Tensor:
        length: int = is_patch_padded.shape[1]
        minimum: float = torch.finfo(is_patch_padded.dtype).min
        padding: Tensor = (is_patch_padded * minimum)[:, None, None, :]
        causal: Tensor = torch.triu(
            torch.full((length, length), minimum, dtype=is_patch_padded.dtype, device=is_patch_padded.device),
            diagonal=1,
        )
        return torch.minimum(padding, causal[None, None])
