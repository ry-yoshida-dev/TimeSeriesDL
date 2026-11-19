from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from .forecast_distribution import MoiraiForecastDistribution
from .model import MoiraiModel

if TYPE_CHECKING:
    from .parameters import MoiraiParameters


class MoiraiForecastNetwork(GenerativeForecastNetwork):
    """
    Moirai over every channel of the window as one multivariate series: the target channels are forecast and the
    other channels are past-only covariates (``past_feat_dynamic_real`` of the original), all in one token sequence.

    Samples come from the predicted mixture, and the training loss is its negative log-likelihood of the normalized
    future values, averaged over the values of each target series and then over the series (the original
    ``PackedNLLLoss`` without the log-scale term, which does not depend on the weights).

    With a fixed ``patch_size`` the whole window is the context. Without one, the patch size is selected per window
    as in the original's ``patch_size="auto"``: the window's last ``input_length - horizon`` steps are the context,
    and each patch size is scored by the likelihood of the window's last ``horizon`` steps given the
    ``input_length - horizon`` steps before them; the training loss then averages over every patch size.

    Parameters
    ----------
    parameters : MoiraiParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast; ``num_parameters`` is the number of sample paths of ``forward``.

    Raises
    ------
    ValueError
        If the patch size is selected automatically but the window is not longer than the horizon.
    """

    def __init__(self, parameters: MoiraiParameters, shape: ForecastShape) -> None:
        if parameters.patch_size is None and shape.input_length <= shape.horizon:
            raise ValueError(
                "selecting the patch size needs input_length > horizon, got "
                + f"{shape.input_length} and {shape.horizon}"
            )
        super().__init__(shape.num_parameters)
        self.model: MoiraiModel = MoiraiModel(parameters)
        self.horizon: int = shape.horizon
        self.patch_size: int | None = parameters.patch_size
        self.patch_sizes: tuple[int, ...] = parameters.patch_sizes
        self.context_length: int = (
            shape.input_length - shape.horizon if parameters.patch_size is None else shape.input_length
        )
        covariates: list[int] = [
            channel for channel in range(shape.input_channels) if channel not in shape.target_channels
        ]
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)
        self.covariate_channels: Tensor
        self.register_buffer("covariate_channels", torch.tensor(covariates, dtype=torch.long), persistent=False)

    @override
    def compute_loss(self, past: Tensor, future_targets: Tensor) -> Tensor:
        targets, covariates = self._split(past[:, -self.context_length :])
        future: Tensor = future_targets.transpose(1, 2)
        sizes: tuple[int, ...] = self.patch_sizes if self.patch_size is None else (self.patch_size,)
        losses: list[Tensor] = [
            self.model(targets, covariates, self.horizon, size).negative_log_likelihood(future).mean() for size in sizes
        ]
        return torch.stack(losses).mean()

    @override
    def sample(self, past: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        targets, covariates = self._split(past[:, -self.context_length :])
        if self.patch_size is not None:
            distribution: MoiraiForecastDistribution = self.model(targets, covariates, self.horizon, self.patch_size)
            return self._draw(distribution, num_samples, generator)
        selected: Tensor = self._select_patch_sizes(past)
        samples: list[Tensor] = []
        for size in self.patch_sizes:
            candidate: MoiraiForecastDistribution = self.model(targets, covariates, self.horizon, size)
            samples.append(self._draw(candidate, num_samples, generator))
        stacked: Tensor = torch.stack(samples)
        return stacked[selected, torch.arange(past.shape[0], device=past.device)]

    def _select_patch_sizes(self, past: Tensor) -> Tensor:
        # The original scores each patch size by the likelihood (in the input scale) of the window's last horizon.
        validation_targets, validation_covariates = self._split(past[:, : self.context_length])
        future: Tensor = self._split(past[:, self.context_length :])[0]
        scores: list[Tensor] = []
        with torch.no_grad():
            for size in self.patch_sizes:
                candidate: MoiraiForecastDistribution = self.model(
                    validation_targets, validation_covariates, self.horizon, size
                )
                loss: Tensor = candidate.negative_log_likelihood(future) + candidate.scale[..., 0, 0].log()
                scores.append(loss.sum(dim=1))
        return torch.stack(scores).argmin(dim=0)

    def _split(self, past: Tensor) -> tuple[Tensor, Tensor]:
        return (
            past.index_select(2, self.target_channels).transpose(1, 2),
            past.index_select(2, self.covariate_channels).transpose(1, 2),
        )

    @staticmethod
    def _draw(distribution: MoiraiForecastDistribution, num_samples: int, generator: torch.Generator | None) -> Tensor:
        if generator is None:
            return distribution.sample(num_samples)
        # The distributions sample from the global generator; seed it from the given one without disturbing it.
        seed: int = int(torch.randint(0, 2**62, (1,), generator=generator, device=generator.device).item())
        with torch.random.fork_rng():
            _ = torch.manual_seed(seed)
            return distribution.sample(num_samples)
