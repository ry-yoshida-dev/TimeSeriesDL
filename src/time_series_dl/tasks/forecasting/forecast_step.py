from dataclasses import dataclass

import torch
from torch import Tensor

from ...models import ForecastNetwork
from .instance_normalization import InstanceNormalization
from .objective import ForecastObjective


@dataclass(frozen=True, eq=False)
class ForecastStep:
    """
    Distribution parameters and loss of a forecast network on scaled windows, with optional instance normalization.

    Attributes
    ----------
    network : ForecastNetwork
        The network.
    objective : ForecastObjective
        The predicted distribution and its loss.
    target_channels : tuple[int, ...]
        Indices of the input channels that are forecast.
    normalization : InstanceNormalization | None
        Normalization of each window by its own statistics; None to feed the scaled values directly.
    """

    network: ForecastNetwork
    objective: ForecastObjective
    target_channels: tuple[int, ...]
    normalization: InstanceNormalization | None

    def compute_parameters(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        """
        Compute the distribution parameters of the horizon.

        Parameters
        ----------
        past : Tensor
            Scaled past values with shape (batch, input_length, input_channels).
        future_targets : Tensor | None
            Scaled true future targets for teacher forcing, given only while training.

        Returns
        -------
        Tensor
            The parameters with shape (batch, horizon, target_channels, num_parameters) in the scale of the inputs.
        """
        if self.normalization is None:
            return self.objective.constrain(self.network(past, future_targets))
        location: Tensor
        scale: Tensor
        location, scale = self.normalization.compute_statistics(past)
        targets: list[int] = list(self.target_channels)
        target_location: Tensor = location[..., targets]
        target_scale: Tensor = scale[..., targets]
        normalized_future: Tensor | None = (
            None if future_targets is None else (future_targets - target_location) / target_scale
        )
        raw: Tensor = self.network((past - location) / scale, normalized_future)
        return self.objective.rescale(self.objective.constrain(raw), target_location, target_scale)

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch of past windows and their true future targets.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled past windows and future targets.
        is_training : bool
            Whether the true targets may be fed to teacher-forced networks.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        past: Tensor = batch[0]
        future: Tensor = batch[1]
        parameters: Tensor = self.compute_parameters(past, future if is_training else None)
        return self.objective.compute_loss(parameters, future)

    def compute_outputs(self, batch: tuple[Tensor, ...]) -> Tensor:
        """
        Compute the point forecast and the quantiles of a batch of past windows.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled past windows first.

        Returns
        -------
        Tensor
            The point forecast stacked before the quantiles, with shape
            (batch, horizon, target_channels, 1 + number of quantile levels).
        """
        parameters: Tensor = self.compute_parameters(batch[0])
        point: Tensor = self.objective.compute_point(parameters).unsqueeze(-1)
        return torch.cat([point, self.objective.compute_quantiles(parameters)], dim=-1)
