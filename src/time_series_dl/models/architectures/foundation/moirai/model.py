from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .distribution_head import MoiraiDistributionHead
from .encoder import MoiraiEncoder
from .forecast_distribution import MoiraiForecastDistribution
from .input_projection import MoiraiInputProjection

if TYPE_CHECKING:
    from .parameters import MoiraiParameters

_MINIMUM_VARIANCE: Final[float] = 1e-5


class MoiraiModel(nn.Module):
    """
    Moirai (Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", ICML 2024): every
    variate is standardized by its own past values and cut into patches; the patches of all variates and the masked
    patches of the target horizon are one token sequence, encoded with any-variate attention, and every forecast
    token is mapped to a mixture distribution of its values.

    Parameters
    ----------
    parameters : MoiraiParameters
        The model configuration.
    """

    def __init__(self, parameters: MoiraiParameters) -> None:
        super().__init__()
        self.patch_sizes: tuple[int, ...] = parameters.patch_sizes
        self.maximum_patch_size: int = max(parameters.patch_sizes)
        self.is_scaled: bool = parameters.is_scaled
        self.mask_embedding: nn.Embedding = nn.Embedding(1, parameters.model_size)
        self.input_projection: MoiraiInputProjection = MoiraiInputProjection(
            parameters.patch_sizes, parameters.model_size
        )
        self.encoder: MoiraiEncoder = MoiraiEncoder(
            parameters.model_size,
            parameters.num_layers,
            parameters.num_heads,
            parameters.feedforward_size,
            parameters.attention_dropout,
            parameters.dropout,
            parameters.layer_norm_epsilon,
        )
        self.distribution_head: MoiraiDistributionHead = MoiraiDistributionHead(
            parameters.model_size, parameters.patch_sizes, parameters.normal_scale
        )

    def forward(self, targets: Tensor, covariates: Tensor, horizon: int, patch_size: int) -> MoiraiForecastDistribution:
        """
        Forecast the distribution of the horizon of the target variates.

        Parameters
        ----------
        targets : Tensor
            Past values of the target variates with shape (batch, targets, length); NaN marks missing values.
        covariates : Tensor
            Past values of the covariate variates with shape (batch, covariates, length), which may have zero
            covariates; NaN marks missing values.
        horizon : int
            Number of future time steps.
        patch_size : int
            One of ``patch_sizes``.

        Returns
        -------
        MoiraiForecastDistribution
            The distribution of the horizon.
        """
        num_targets: int = targets.shape[1]
        values: Tensor = torch.cat([targets, covariates], dim=1).to(torch.float32)
        is_observed: Tensor = ~torch.isnan(values)
        values = torch.where(is_observed, values, 0.0)
        location, scale = self._compute_statistics(values, is_observed)
        patches: Tensor = self._patch(values, patch_size)
        is_patch_observed: Tensor = self._patch(is_observed.to(values.dtype), patch_size) > 0
        past_tokens: int = patches.shape[2]
        forecast_tokens: int = math.ceil(horizon / patch_size)
        batch: int = values.shape[0]
        device: torch.device = values.device
        embedded: Tensor = self.input_projection((patches - location) / scale, patch_size)
        masked: Tensor = self.mask_embedding.weight.expand(batch, num_targets * forecast_tokens, -1)
        tokens: Tensor = torch.cat(
            [embedded[:, :num_targets].flatten(1, 2), masked, embedded[:, num_targets:].flatten(1, 2)], dim=1
        )
        past_time: Tensor = self._compute_time_ids(is_patch_observed[:, :num_targets])
        forecast_time: Tensor = (
            torch.arange(forecast_tokens, device=device)[None, :] + past_time.max(dim=-1, keepdim=True).values + 1
        )
        num_covariates: int = values.shape[1] - num_targets
        time_ids: Tensor = torch.cat(
            [
                past_time.repeat(1, num_targets),
                forecast_time.repeat(1, num_targets),
                past_time.repeat(1, num_covariates),
            ],
            dim=1,
        )
        variates: Tensor = torch.arange(values.shape[1], device=device)
        variate_ids: Tensor = torch.cat(
            [
                variates[:num_targets].repeat_interleave(past_tokens),
                variates[:num_targets].repeat_interleave(forecast_tokens),
                variates[num_targets:].repeat_interleave(past_tokens),
            ]
        ).expand(batch, -1)
        encoded: Tensor = self.encoder(tokens, variate_ids, time_ids)
        start: int = num_targets * past_tokens
        forecast: Tensor = encoded[:, start : start + num_targets * forecast_tokens].reshape(
            batch, num_targets, forecast_tokens, -1
        )
        return MoiraiForecastDistribution(
            mixture=self.distribution_head(forecast, patch_size),
            location=location[:, :num_targets],
            scale=scale[:, :num_targets],
            patch_size=patch_size,
            horizon=horizon,
        )

    def __call__(
        self, targets: Tensor, covariates: Tensor, horizon: int, patch_size: int
    ) -> MoiraiForecastDistribution:
        return cast(MoiraiForecastDistribution, super().__call__(targets, covariates, horizon, patch_size))

    def _compute_statistics(self, values: Tensor, is_observed: Tensor) -> tuple[Tensor, Tensor]:
        if not self.is_scaled:
            zeros: Tensor = torch.zeros(values.shape[0], values.shape[1], 1, 1, device=values.device)
            return zeros, torch.ones_like(zeros)
        # As the original, in double precision (single on Apple MPS, which has none) with the unbiased variance and
        # a floor added to it.
        precision: torch.dtype = torch.float32 if values.device.type == "mps" else torch.float64
        doubled: Tensor = values.to(precision)
        count: Tensor = is_observed.sum(dim=-1, keepdim=True).to(precision)
        location: Tensor = (doubled * is_observed).sum(dim=-1, keepdim=True) / torch.where(count == 0, 1.0, count)
        squares: Tensor = ((doubled - location).square() * is_observed).sum(dim=-1, keepdim=True)
        variance: Tensor = squares / torch.where(count - 1 == 0, 1.0, count - 1)
        scale: Tensor = torch.sqrt(variance + _MINIMUM_VARIANCE)
        return location.to(torch.float32)[..., None], scale.to(torch.float32)[..., None]

    def _patch(self, values: Tensor, patch_size: int) -> Tensor:
        # Left padding to whole patches (unobserved zeros), then zeros up to the largest patch size.
        padded: Tensor = functional.pad(values, (-values.shape[-1] % patch_size, 0))
        patches: Tensor = padded.reshape(*padded.shape[:-1], -1, patch_size)
        return functional.pad(patches, (0, self.maximum_patch_size - patch_size))

    @staticmethod
    def _compute_time_ids(is_target_observed: Tensor) -> Tensor:
        # A token's time index counts the patches since the first one with an observed target value.
        is_token_observed: Tensor = is_target_observed.any(dim=-1).any(dim=1).to(torch.long)
        return (is_token_observed.cummax(dim=-1).values.cumsum(dim=-1) - 1).clamp(min=0)
