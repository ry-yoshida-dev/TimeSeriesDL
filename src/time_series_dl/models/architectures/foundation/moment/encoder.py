from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.t5 import T5Encoder
from .encoding import MOMENTEncoding
from .patch_embedding import MOMENTPatchEmbedding

if TYPE_CHECKING:
    from .parameters import MOMENTParameters


class MOMENTEncoder(nn.Module):
    """
    MOMENT's encoder of univariate series of a fixed length: reversible instance normalization by the mean and
    standard deviation of the observed values (without affine parameters), non-overlapping patches, the patch
    embedding, and a T5 encoder whose attention skips patches that are not entirely part of the input.

    Shorter series are left-padded with zeros by ``pad``, which marks the padding as outside the input, as the
    official code expects of its callers.

    Parameters
    ----------
    parameters : MOMENTParameters
        The model configuration.
    """

    def __init__(self, parameters: MOMENTParameters) -> None:
        super().__init__()
        self.input_length: int = parameters.input_length
        self.patch_length: int = parameters.patch_length
        self.epsilon: float = parameters.normalization_epsilon
        self.patch_embedding: MOMENTPatchEmbedding = MOMENTPatchEmbedding(
            parameters.patch_length,
            parameters.model_size,
            parameters.patch_dropout,
            parameters.is_positional_embedding_used,
            parameters.num_patches,
        )
        self.transformer: T5Encoder = T5Encoder(parameters.encoder_configuration)
        if parameters.is_backbone_frozen:
            for parameter in self.parameters():
                _ = parameter.requires_grad_(False)

    def pad(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """
        Left-pad series to the input length.

        Parameters
        ----------
        values : Tensor
            Series with shape (batch, length), at most ``input_length`` long.

        Returns
        -------
        tuple[Tensor, Tensor]
            The padded series with shape (batch, input_length) and the boolean mask of the original values.
        """
        padding: int = self.input_length - values.shape[1]
        is_input: Tensor = torch.ones_like(values, dtype=torch.bool)
        return functional.pad(values, (padding, 0), value=0.0), functional.pad(is_input, (padding, 0), value=False)

    def expand_whole_patches(self, mask: Tensor) -> Tensor:
        """
        Keep only the values of patches whose values are all set in a mask.

        Parameters
        ----------
        mask : Tensor
            Boolean mask with shape (batch, input_length).

        Returns
        -------
        Tensor
            The mask, False throughout every patch that has a False value.
        """
        return self._patch(mask).all(dim=-1).repeat_interleave(self.patch_length, dim=-1)

    def forward(self, values: Tensor, is_observed: Tensor, is_visible: Tensor, is_input: Tensor) -> MOMENTEncoding:
        """
        Encode a batch of series.

        Parameters
        ----------
        values : Tensor
            Series with shape (batch, input_length).
        is_observed : Tensor
            Boolean mask of the values the normalization statistics are computed from.
        is_visible : Tensor
            Boolean mask of the values the network may see; a patch with any hidden value is replaced by the mask
            embedding.
        is_input : Tensor
            Boolean mask of the values that are part of the series rather than padding; only patches entirely
            inside the input are attended to.

        Returns
        -------
        MOMENTEncoding
            The patch states and the normalization statistics.
        """
        count: Tensor = is_observed.sum(dim=-1, keepdim=True)
        divisor: Tensor = count.clamp(min=1)
        location: Tensor = torch.where(is_observed, values, 0.0).sum(dim=-1, keepdim=True) / divisor
        variance: Tensor = torch.where(is_observed, (values - location).square(), 0.0).sum(dim=-1, keepdim=True)
        # Series without observed values get location 0 and scale 1 where the original would produce NaN.
        location = torch.where(count > 0, location, 0.0)
        scale: Tensor = torch.where(count > 0, (variance / divisor).sqrt() + self.epsilon, 1.0)
        normalized: Tensor = torch.nan_to_num((values - location) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        patches: Tensor = self._patch(normalized)
        embeddings: Tensor = self.patch_embedding(patches, self._patch(is_visible).all(dim=-1))
        states: Tensor = self.transformer(embeddings, self._patch(is_input).all(dim=-1))
        return MOMENTEncoding(states=states, location=location, scale=scale)

    def __call__(self, values: Tensor, is_observed: Tensor, is_visible: Tensor, is_input: Tensor) -> MOMENTEncoding:
        return cast(MOMENTEncoding, super().__call__(values, is_observed, is_visible, is_input))

    def _patch(self, values: Tensor) -> Tensor:
        return values.unfold(dimension=-1, size=self.patch_length, step=self.patch_length)
