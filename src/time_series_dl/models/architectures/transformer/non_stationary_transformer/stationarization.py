from dataclasses import dataclass
from typing import ClassVar, Self

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True, eq=False)
class Stationarization:
    """
    Series stationarization of the Non-stationary Transformer: each window is normalized by the mean and standard
    deviation of its own values, which are kept to restore the outputs and to learn the de-stationary factors.

    Attributes
    ----------
    mean : Tensor
        The mean of each window and channel with shape (batch, 1, channels), detached from the graph.
    standard_deviation : Tensor
        The standard deviation of each window and channel with shape (batch, 1, channels), detached from the graph.
    """

    mean: Tensor
    standard_deviation: Tensor

    EPSILON: ClassVar[float] = 1e-5

    @classmethod
    def from_values(cls, values: Tensor) -> Self:
        """
        Compute the statistics of a batch of windows.

        The variance is the biased one, and ``EPSILON`` is added to it before taking the square root.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Stationarization
            The statistics.
        """
        mean: Tensor = values.mean(dim=1, keepdim=True).detach()
        variance: Tensor = values.var(dim=1, keepdim=True, unbiased=False).detach()
        return cls(mean=mean, standard_deviation=torch.sqrt(variance + cls.EPSILON))

    def normalize(self, values: Tensor) -> Tensor:
        """
        Normalize a batch of windows.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Values with zero mean and unit standard deviation per window and channel.
        """
        return (values - self.mean) / self.standard_deviation

    def restore(self, values: Tensor) -> Tensor:
        """
        Map normalized values back to the scale of the windows.

        Parameters
        ----------
        values : Tensor
            Normalized values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Values in the scale of the windows.
        """
        return values * self.standard_deviation + self.mean
