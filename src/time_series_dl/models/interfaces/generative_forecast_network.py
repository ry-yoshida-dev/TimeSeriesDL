from abc import abstractmethod
from typing import override

import torch
from torch import Tensor

from .forecast_network import ForecastNetwork


class GenerativeForecastNetwork(ForecastNetwork):
    """
    A forecast network that draws sample paths of the horizon and is trained with its own loss, such as the
    denoising loss of a diffusion model, instead of a likelihood of per-value distribution parameters.

    As a ``ForecastNetwork`` it outputs ``num_samples`` sample paths in place of the distribution parameters, so every
    value's "parameters" are its samples; the forecasting task reads them with ``SampleObjective`` and trains the
    network with ``compute_loss``.

    Parameters
    ----------
    num_samples : int
        Number of sample paths that ``forward`` draws; the ``num_parameters`` of the forecast shape.

    Raises
    ------
    ValueError
        If num_samples is not positive.
    """

    def __init__(self, num_samples: int) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        super().__init__()
        self.num_samples: int = num_samples

    @abstractmethod
    def compute_loss(self, past: Tensor, future_targets: Tensor) -> Tensor:
        """
        Compute the training loss of a batch of windows.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, input_length, input_channels).
        future_targets : Tensor
            True future target values with shape (batch, horizon, output_channels).

        Returns
        -------
        Tensor
            The scalar loss.
        """

    @abstractmethod
    def sample(self, past: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        """
        Draw sample paths of the horizon of a batch of windows.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, input_length, input_channels).
        num_samples : int
            Number of sample paths per window.
        generator : torch.Generator | None
            Source of randomness on the device of the network; PyTorch's global generator when None.

        Returns
        -------
        Tensor
            The samples with shape (batch, num_samples, horizon, output_channels).
        """

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        """
        Draw ``num_samples`` sample paths of the horizon, as distribution parameters.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, input_length, input_channels).
        future_targets : Tensor | None
            Ignored: training uses ``compute_loss``.

        Returns
        -------
        Tensor
            The samples with shape (batch, horizon, output_channels, num_samples).
        """
        return self.sample(past, self.num_samples).permute(0, 2, 3, 1)
