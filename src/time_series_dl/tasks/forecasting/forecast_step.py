from dataclasses import dataclass

import torch
from torch import Tensor

from ...models.interfaces import CovariateForecastNetwork, CovariateTensors, ForecastNetwork
from ...models.interfaces.generative_forecast_network import GenerativeForecastNetwork
from .instance_normalization import InstanceNormalization
from .normalized_window import NormalizedWindow
from .objective import ForecastObjective


@dataclass(frozen=True, eq=False)
class ForecastStep:
    """
    Distribution parameters, samples and loss of a forecast network on scaled windows, with optional instance
    normalization.

    A batch holds the past windows, then, for a covariate network, the known covariates, the numeric static features
    and the categorical static codes, and, for the loss, the future targets last. Instance normalization applies to the
    past values only: covariates keep their global scaling, and covariate networks receive the location and the log
    scale of each target channel as extra numeric static features.

    A ``GenerativeForecastNetwork`` is trained with its own loss on the normalized windows, and its parameters are
    sample paths it draws; every other network is trained with the objective's loss.

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

    def compute_parameters(
        self,
        inputs: tuple[Tensor, ...],
        future_targets: Tensor | None = None,
        num_samples: int = 0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """
        Compute the distribution parameters of the horizon.

        Parameters
        ----------
        inputs : tuple[Tensor, ...]
            Scaled past values with shape (batch, input_length, input_channels), followed for a covariate network by
            the scaled known covariates, numeric static features and categorical static codes.
        future_targets : Tensor | None
            Scaled true future targets for teacher forcing, given only while training.
        num_samples : int
            Number of sample paths a generative network draws as parameters; 0 for the objective's
            ``num_parameters``. Other networks ignore it.
        generator : torch.Generator | None
            Source of randomness of a generative network; PyTorch's global generator when None.

        Returns
        -------
        Tensor
            The parameters with shape (batch, horizon, target_channels, num_parameters) in the scale of the inputs.
        """
        window: NormalizedWindow = self._normalize(inputs[0], future_targets)
        raw: Tensor = self._run(window, inputs[1:], num_samples, generator)
        constrained: Tensor = self.objective.constrain(raw)
        if self.normalization is None:
            return constrained
        if isinstance(self.network, GenerativeForecastNetwork):
            # Any number of samples may be drawn, and every sample is in the scale of the values.
            return constrained * window.target_scale.unsqueeze(-1) + window.target_location.unsqueeze(-1)
        return self.objective.rescale(constrained, window.target_location, window.target_scale)

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
        if isinstance(self.network, GenerativeForecastNetwork):
            window: NormalizedWindow = self._normalize(batch[0], future)
            normalized_future: Tensor = future if window.future_targets is None else window.future_targets
            return self.network.compute_loss(window.past, normalized_future)
        parameters: Tensor = self.compute_parameters(batch[:-1], future if is_training else None)
        return self.objective.compute_loss(parameters, future)

    def compute_outputs(
        self, batch: tuple[Tensor, ...], num_samples: int = 0, generator: torch.Generator | None = None
    ) -> Tensor:
        """
        Compute the point forecast, the quantiles and optionally sample paths of a batch of past windows.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled inputs of ``compute_parameters``.
        num_samples : int
            Number of sample paths to draw; 0 for none. A generative network draws exactly these paths (or the
            objective's ``num_parameters`` when 0), and its point forecast and quantiles are statistics of them.
        generator : torch.Generator | None
            Source of randomness of the samples; PyTorch's global generator when None.

        Returns
        -------
        Tensor
            The point forecast, then the quantiles, then the samples, stacked along the last dimension, with shape
            (batch, horizon, target_channels, 1 + number of quantile levels + num_samples).

        Raises
        ------
        ValueError
            If num_samples is negative.
        """
        if num_samples < 0:
            raise ValueError(f"num_samples must not be negative, got {num_samples}")
        parameters: Tensor = self.compute_parameters(batch, None, num_samples, generator)
        outputs: list[Tensor] = [
            self.objective.compute_point(parameters).unsqueeze(-1),
            self.objective.compute_quantiles(parameters),
        ]
        if num_samples > 0:
            outputs.append(
                parameters
                if isinstance(self.network, GenerativeForecastNetwork)
                else self.objective.sample(parameters, num_samples, generator).permute(0, 2, 3, 1)
            )
        return torch.cat(outputs, dim=-1)

    def _normalize(self, past: Tensor, future_targets: Tensor | None) -> NormalizedWindow:
        if self.normalization is None:
            return NormalizedWindow(past=past, future_targets=future_targets, statistics=None)
        location: Tensor
        scale: Tensor
        location, scale = self.normalization.compute_statistics(past)
        targets: list[int] = list(self.target_channels)
        target_location: Tensor = location[..., targets]
        target_scale: Tensor = scale[..., targets]
        return NormalizedWindow(
            past=(past - location) / scale,
            future_targets=None if future_targets is None else (future_targets - target_location) / target_scale,
            statistics=(target_location, target_scale),
        )

    def _run(
        self,
        window: NormalizedWindow,
        covariate_inputs: tuple[Tensor, ...],
        num_samples: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        match self.network:
            case CovariateForecastNetwork() as network:
                known: Tensor
                static: Tensor
                categories: Tensor
                known, static, categories = covariate_inputs
                if window.statistics is not None:
                    static = torch.cat([static, window.static_features], dim=-1)
                return network(
                    window.past,
                    CovariateTensors(known=known, static=static, static_categories=categories),
                    window.future_targets,
                )
            case GenerativeForecastNetwork() as network:
                count: int = num_samples if num_samples > 0 else self.objective.num_parameters
                return network.sample(window.past, count, generator).permute(0, 2, 3, 1)
            case ForecastNetwork() as network:
                return network(window.past, window.future_targets)
