from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn


class ForecastNetwork(nn.Module, ABC):
    """
    Maps a window of past values to the distribution parameters of the future target values.
    """

    @abstractmethod
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        """
        Forecast the horizon of a batch of windows.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, input_length, input_channels).
        future_targets : Tensor | None
            True future target values with shape (batch, horizon, output_channels), given only while training, for
            networks that use teacher forcing; every other network ignores them.

        Returns
        -------
        Tensor
            Unconstrained distribution parameters with shape (batch, horizon, output_channels, num_parameters).
        """

    def __call__(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        return cast(Tensor, super().__call__(past, future_targets))
