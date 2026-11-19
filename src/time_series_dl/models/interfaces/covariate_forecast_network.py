from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn

from .covariate_tensors import CovariateTensors


class CovariateForecastNetwork(nn.Module, ABC):
    """
    Maps a window of past values and the covariates of the window to the distribution parameters of the future
    target values.

    The covariates are scaled by the task but never normalized by the statistics of the window, so their values mean
    the same in every window.
    """

    @abstractmethod
    def forward(self, past: Tensor, covariates: CovariateTensors, future_targets: Tensor | None = None) -> Tensor:
        """
        Forecast the horizon of a batch of windows.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, input_length, input_channels).
        covariates : CovariateTensors
            Known covariates of the past window and the horizon, and static features of each window's series.
        future_targets : Tensor | None
            True future target values with shape (batch, horizon, output_channels), given only while training, for
            networks that use teacher forcing; every other network ignores them.

        Returns
        -------
        Tensor
            Unconstrained distribution parameters with shape (batch, horizon, output_channels, num_parameters).
        """

    def __call__(self, past: Tensor, covariates: CovariateTensors, future_targets: Tensor | None = None) -> Tensor:
        return cast(Tensor, super().__call__(past, covariates, future_targets))
