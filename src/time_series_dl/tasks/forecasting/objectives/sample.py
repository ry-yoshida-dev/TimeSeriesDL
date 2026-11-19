from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from ..objective import ForecastObjective
from ..sample_point_estimate import SamplePointEstimate


@dataclass(frozen=True)
class SampleObjective(ForecastObjective):
    """
    An ensemble of sample paths in place of distribution parameters: every forecast value's parameters are its
    ``num_samples`` samples, and point forecasts and quantiles are statistics of the samples.

    Generative networks (TimeGrad, CSDI), which draw the samples and are trained with their own loss, require this
    objective. Any other network may also use it: it then outputs the ensemble directly and is trained with the fair
    (unbiased) continuous ranked probability score of the ensemble (Ferro, 2014), which rewards a calibrated spread.

    Attributes
    ----------
    num_samples : int
        Number of sample paths per window; at least 2.
    prediction_quantiles : tuple[float, ...]
        Levels of the quantiles that predictions report, strictly increasing in (0, 1).
    point_estimate : SamplePointEstimate
        Statistic of the samples that serves as the point forecast.

    Raises
    ------
    ValueError
        If num_samples is below 2, or the levels are out of (0, 1) or not strictly increasing.
    """

    num_samples: int = 100
    prediction_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    point_estimate: SamplePointEstimate = SamplePointEstimate.MEDIAN

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError(f"num_samples must be at least 2, got {self.num_samples}")
        if any(not 0.0 < level < 1.0 for level in self.prediction_quantiles):
            raise ValueError(f"prediction_quantiles must be in (0, 1), got {self.prediction_quantiles}")
        if any(
            lower >= upper
            for lower, upper in zip(self.prediction_quantiles, self.prediction_quantiles[1:], strict=False)
        ):
            raise ValueError(f"prediction_quantiles must be strictly increasing, got {self.prediction_quantiles}")

    @property
    @override
    def num_parameters(self) -> int:
        return self.num_samples

    @property
    @override
    def location_index(self) -> int:
        return 0

    @property
    @override
    def level_parameters(self) -> tuple[int, ...]:
        return tuple(range(self.num_samples))

    @property
    @override
    def scale_parameters(self) -> tuple[int, ...]:
        return ()

    @property
    @override
    def quantile_levels(self) -> tuple[float, ...]:
        return self.prediction_quantiles

    @override
    def constrain(self, raw: Tensor) -> Tensor:
        return raw

    @override
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        count: int = parameters.shape[-1]
        if count < 2:
            raise ValueError(f"the fair CRPS needs at least 2 samples, got {count}")
        error: Tensor = (parameters - targets.unsqueeze(-1)).abs().mean(dim=-1)
        # sum_{i,j} |x_i - x_j| = 2 sum_k (2k - m - 1) x_(k) over the sorted samples x_(1) <= ... <= x_(m).
        ordered: Tensor = parameters.sort(dim=-1).values
        ranks: Tensor = torch.arange(1, count + 1, device=parameters.device, dtype=parameters.dtype)
        spread: Tensor = (ordered * (2.0 * ranks - count - 1.0)).sum(dim=-1) / (count * (count - 1))
        return (error - spread).mean()

    @override
    def compute_point(self, parameters: Tensor) -> Tensor:
        return self.point_estimate.compute(parameters)

    @override
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        count: int = parameters.shape[-1]
        ordered: Tensor = parameters.sort(dim=-1).values
        # Linear interpolation between order statistics, as numpy.quantile does by default.
        positions: Tensor = torch.tensor(
            self.prediction_quantiles, device=parameters.device, dtype=parameters.dtype
        ) * (count - 1)
        lower: Tensor = positions.floor().long().clamp(0, count - 1)
        upper: Tensor = (lower + 1).clamp(max=count - 1)
        weight: Tensor = positions - lower.to(parameters.dtype)
        index_shape: tuple[int, ...] = (*parameters.shape[:-1], len(self.prediction_quantiles))
        lower_values: Tensor = ordered.gather(-1, lower.expand(index_shape))
        upper_values: Tensor = ordered.gather(-1, upper.expand(index_shape))
        return lower_values + weight * (upper_values - lower_values)

    @override
    def sample(self, parameters: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        # Whole paths are resampled, so the dependence across horizon steps and channels is kept.
        members: Tensor = torch.randint(
            0, parameters.shape[-1], (parameters.shape[0], num_samples), generator=generator, device=parameters.device
        )
        index: Tensor = members[:, None, None, :].expand(*parameters.shape[:-1], num_samples)
        return parameters.gather(-1, index).permute(0, 3, 1, 2)
