from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from .mixture import MoiraiMixture


@dataclass(frozen=True, eq=False)
class MoiraiForecastDistribution:
    """
    Moirai's predictive distribution of the horizon of every target variate: the normalized mixture of every value
    of the forecast patches, and the location and scale that map normalized values back.

    Attributes
    ----------
    mixture : MoiraiMixture
        Normalized distribution with values of shape (batch, targets, forecast_tokens, max(patch_sizes)); only the
        first ``patch_size`` values of each token are forecasts.
    location : Tensor
        Location of each target variate with shape (batch, targets, 1, 1).
    scale : Tensor
        Scale of each target variate with shape (batch, targets, 1, 1).
    patch_size : int
        The patch size of the tokens.
    horizon : int
        Number of forecast time steps.
    """

    mixture: MoiraiMixture
    location: Tensor
    scale: Tensor
    patch_size: int
    horizon: int

    def sample(self, num_samples: int) -> Tensor:
        """
        Draw sample paths in the scale of the inputs with PyTorch's global random number generator.

        Parameters
        ----------
        num_samples : int
            Number of paths.

        Returns
        -------
        Tensor
            Samples with shape (batch, num_samples, horizon, targets).
        """
        normalized: Tensor = self.mixture.sample(num_samples)[..., : self.patch_size]
        paths: Tensor = normalized.flatten(start_dim=-2)[..., : self.horizon]
        restored: Tensor = paths * self.scale[..., 0] + self.location[..., 0]
        return restored.permute(1, 0, 3, 2)

    def negative_log_likelihood(self, future: Tensor) -> Tensor:
        """
        Compute the mean negative log density of each target variate's observed future values, normalized by the
        variate's location and scale (the log-scale term of the density in the input scale is left out).

        Parameters
        ----------
        future : Tensor
            True future values with shape (batch, targets, horizon) in the scale of the inputs; NaN marks missing
            values.

        Returns
        -------
        Tensor
            The mean negative log density with shape (batch, targets); 0 for variates without observed values.
        """
        tokens: int = self.mixture.student_t_location.shape[2]
        width: int = self.mixture.student_t_location.shape[3]
        is_observed: Tensor = ~torch.isnan(future)
        normalized: Tensor = (torch.where(is_observed, future, 0.0) - self.location[..., 0]) / self.scale[..., 0]
        padding: int = tokens * self.patch_size - self.horizon
        values: Tensor = functional.pad(normalized, (0, padding)).reshape(
            *normalized.shape[:-1], tokens, self.patch_size
        )
        mask: Tensor = functional.pad(is_observed.to(normalized.dtype), (0, padding)).reshape(
            *normalized.shape[:-1], tokens, self.patch_size
        )
        values = functional.pad(values, (0, width - self.patch_size))
        mask = functional.pad(mask, (0, width - self.patch_size))
        log_density: Tensor = torch.where(mask > 0, self.mixture.log_prob(values), 0.0)
        count: Tensor = mask.sum(dim=(-1, -2))
        return -log_density.sum(dim=(-1, -2)) / count.clamp(min=1)
