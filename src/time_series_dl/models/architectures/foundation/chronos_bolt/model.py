from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.t5 import T5Decoder, T5Encoder
from .residual_block import ChronosBoltResidualBlock

if TYPE_CHECKING:
    from .parameters import ChronosBoltParameters

_DECODER_START_TOKEN: Final[int] = 0
_REGISTER_TOKEN: Final[int] = 1
_SCALE_EPSILON: Final[float] = 1e-5


class ChronosBoltModel(nn.Module):
    """
    Chronos-Bolt (Ansari et al., 2024): each series is standardized by its own mean and standard deviation, cut into
    patches that a residual block embeds together with their observation mask, encoded by a T5 encoder (with a
    trailing register token), and a single decoder step turns the encoder states into the quantiles of the next
    ``prediction_length`` values at once.

    Parameters
    ----------
    parameters : ChronosBoltParameters
        The model configuration.
    """

    def __init__(self, parameters: ChronosBoltParameters) -> None:
        super().__init__()
        self.context_length: int = parameters.context_length
        self.patch_length: int = parameters.patch_length
        self.patch_stride: int = parameters.patch_stride
        self.prediction_length: int = parameters.prediction_length
        self.num_quantiles: int = len(parameters.quantile_levels)
        self.is_register_token_used: bool = parameters.is_register_token_used
        self.token_embedding: nn.Embedding = nn.Embedding(
            2 if parameters.is_register_token_used else 1, parameters.model_size
        )
        self.input_embedding: ChronosBoltResidualBlock = ChronosBoltResidualBlock(
            2 * parameters.patch_length, parameters.feedforward_size, parameters.model_size, parameters.dropout
        )
        self.encoder: T5Encoder = T5Encoder(parameters.encoder_configuration)
        self.decoder: T5Decoder = T5Decoder(parameters.decoder_configuration)
        self.output_projection: ChronosBoltResidualBlock = ChronosBoltResidualBlock(
            parameters.model_size,
            parameters.feedforward_size,
            self.num_quantiles * parameters.prediction_length,
            parameters.dropout,
        )

    def forward(self, context: Tensor) -> Tensor:
        """
        Forecast the quantiles of the next ``prediction_length`` values of each series.

        Parameters
        ----------
        context : Tensor
            Past values with shape (batch, length); NaN marks missing values. Only the last ``context_length`` values
            are read.

        Returns
        -------
        Tensor
            Quantiles with shape (batch, num_quantiles, prediction_length) in the scale of the context.
        """
        context = context[:, -self.context_length :].to(torch.float32)
        is_observed: Tensor = ~torch.isnan(context)
        location, scale = self._compute_statistics(context, is_observed)
        scaled: Tensor = torch.where(is_observed, (context - location) / scale, 0.0)
        patches: Tensor = self._patch(scaled)
        patch_mask: Tensor = self._patch(is_observed.to(torch.float32))
        embeddings: Tensor = self.input_embedding(torch.cat([patches, patch_mask], dim=-1))
        is_patch_observed: Tensor = patch_mask.sum(dim=-1) > 0
        batch: int = context.shape[0]
        if self.is_register_token_used:
            register: Tensor = self.token_embedding.weight[_REGISTER_TOKEN].expand(batch, 1, -1)
            embeddings = torch.cat([embeddings, register], dim=1)
            is_patch_observed = functional.pad(is_patch_observed, (0, 1), value=True)
        encoded: Tensor = self.encoder(embeddings, is_patch_observed)
        start: Tensor = self.token_embedding.weight[_DECODER_START_TOKEN].expand(batch, 1, -1)
        decoded: Tensor = self.decoder(start, encoded, is_patch_observed)
        quantiles: Tensor = self.output_projection(decoded[:, 0]).reshape(
            batch, self.num_quantiles, self.prediction_length
        )
        return quantiles * scale.unsqueeze(-1) + location.unsqueeze(-1)

    def __call__(self, context: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(context))

    @staticmethod
    def _compute_statistics(context: Tensor, is_observed: Tensor) -> tuple[Tensor, Tensor]:
        count: Tensor = is_observed.sum(dim=-1, keepdim=True)
        observed: Tensor = torch.where(is_observed, context, 0.0)
        # Series without any observation get location 0 and scale 1, like the NaN replacements of the original.
        location: Tensor = torch.where(count > 0, observed.sum(dim=-1, keepdim=True) / count.clamp(min=1), 0.0)
        deviation: Tensor = torch.where(is_observed, context - location, 0.0)
        variance: Tensor = deviation.square().sum(dim=-1, keepdim=True) / count.clamp(min=1)
        scale: Tensor = torch.where(count > 0, variance.sqrt(), 1.0)
        return location, torch.where(scale == 0, _SCALE_EPSILON, scale)

    def _patch(self, values: Tensor) -> Tensor:
        remainder: int = values.shape[-1] % self.patch_length
        if remainder != 0:
            # Left padding counts as unobserved: zero values with a zero mask.
            values = functional.pad(values, (self.patch_length - remainder, 0), value=0.0)
        return values.unfold(dimension=-1, size=self.patch_length, step=self.patch_stride)
