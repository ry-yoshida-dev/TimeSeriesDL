from dataclasses import dataclass

import torch
from torch import Tensor

from ...models.interfaces import CovariateForecastNetwork, CovariateTensors, ForecastNetwork
from .instance_normalization import InstanceNormalization
from .objective import ForecastObjective


@dataclass(frozen=True, eq=False)
class ForecastStep:
    """
    Distribution parameters and loss of a forecast network on scaled windows, with optional instance normalization.

    A batch holds the past windows, then, for a covariate network, the known covariates, the numeric static features
    and the categorical static codes, and, for the loss, the future targets last. Instance normalization applies to the
    past values only: covariates keep their global scaling, and covariate networks receive the location and the log
    scale of each target channel as extra numeric static features.

    Attributes
    ----------
    network : ForecastNetwork | CovariateForecastNetwork
        The network.
    objective : ForecastObjective
        The predicted distribution and its loss.
    target_channels : tuple[int, ...]
        Indices of the input channels that are forecast.
    normalization : InstanceNormalization | None
        Normalization of each window by its own statistics; None to feed the scaled values directly.
    """

    network: ForecastNetwork | CovariateForecastNetwork
    objective: ForecastObjective
    target_channels: tuple[int, ...]
    normalization: InstanceNormalization | None

    def compute_parameters(self, inputs: tuple[Tensor, ...], future_targets: Tensor | None = None) -> Tensor:
        """
        Compute the distribution parameters of the horizon.

        Parameters
        ----------
        inputs : tuple[Tensor, ...]
            Scaled past values with shape (batch, input_length, input_channels), followed for a covariate network by
            the scaled known covariates, numeric static features and categorical static codes.
        future_targets : Tensor | None
            Scaled true future targets for teacher forcing, given only while training.

        Returns
        -------
        Tensor
            The parameters with shape (batch, horizon, target_channels, num_parameters) in the scale of the inputs.
        """
        past: Tensor = inputs[0]
        if self.normalization is None:
            return self.objective.constrain(self._run(past, inputs[1:], None, future_targets))
        location: Tensor
        scale: Tensor
        location, scale = self.normalization.compute_statistics(past)
        targets: list[int] = list(self.target_channels)
        target_location: Tensor = location[..., targets]
        target_scale: Tensor = scale[..., targets]
        normalized_future: Tensor | None = (
            None if future_targets is None else (future_targets - target_location) / target_scale
        )
        statistics: Tensor = torch.cat([target_location, target_scale.log()], dim=-1).squeeze(1)
        raw: Tensor = self._run((past - location) / scale, inputs[1:], statistics, normalized_future)
        return self.objective.rescale(self.objective.constrain(raw), target_location, target_scale)

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch of past windows and their true future targets.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled inputs of ``compute_parameters`` followed by the future targets.
        is_training : bool
            Whether the true targets may be fed to teacher-forced networks.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        future: Tensor = batch[-1]
        parameters: Tensor = self.compute_parameters(batch[:-1], future if is_training else None)
        return self.objective.compute_loss(parameters, future)

    def compute_outputs(self, batch: tuple[Tensor, ...]) -> Tensor:
        """
        Compute the point forecast and the quantiles of a batch of past windows.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled inputs of ``compute_parameters``.

        Returns
        -------
        Tensor
            The point forecast stacked before the quantiles, with shape
            (batch, horizon, target_channels, 1 + number of quantile levels).
        """
        parameters: Tensor = self.compute_parameters(batch)
        point: Tensor = self.objective.compute_point(parameters).unsqueeze(-1)
        return torch.cat([point, self.objective.compute_quantiles(parameters)], dim=-1)

    def _run(
        self,
        past: Tensor,
        covariate_inputs: tuple[Tensor, ...],
        window_statistics: Tensor | None,
        future_targets: Tensor | None,
    ) -> Tensor:
        match self.network:
            case CovariateForecastNetwork() as network:
                known: Tensor
                static: Tensor
                categories: Tensor
                known, static, categories = covariate_inputs
                if window_statistics is not None:
                    static = torch.cat([static, window_statistics], dim=-1)
                return network(
                    past, CovariateTensors(known=known, static=static, static_categories=categories), future_targets
                )
            case ForecastNetwork() as network:
                return network(past, future_targets)
