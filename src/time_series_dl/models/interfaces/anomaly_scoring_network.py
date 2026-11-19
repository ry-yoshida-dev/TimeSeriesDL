from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn


class AnomalyScoringNetwork(nn.Module, ABC):
    """
    Anomaly detector with its own training objective and its own anomaly score of each time step, such as the
    association discrepancy of the Anomaly Transformer or the adversarial reconstruction errors of USAD.

    Training minimizes ``compute_loss`` on windows of normal behavior with one optimizer over all parameters; models
    trained in alternating or adversarial phases express them as one combined loss whose gradients equal those of the
    paper's phases, and may change the loss with the epoch.
    """

    @abstractmethod
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        """
        Compute the training loss of a batch of windows.

        Parameters
        ----------
        values : Tensor
            Scaled windows of normal behavior with shape (batch, length, channels).
        epoch : int
            Zero-based index of the current training epoch, for losses whose phase weights change during training.

        Returns
        -------
        Tensor
            The scalar loss.
        """

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """
        Score every time step of a batch of windows.

        Parameters
        ----------
        values : Tensor
            Scaled windows with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Anomaly scores with shape (batch, length); higher is more anomalous. NaN marks steps the network does not
            score, such as the past window of a forecast.
        """

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
