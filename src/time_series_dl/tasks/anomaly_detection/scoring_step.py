from dataclasses import dataclass, field

from torch import Tensor

from ...models import ForecastNetwork, ReconstructionNetwork
from ...models.interfaces.anomaly_scoring_network import AnomalyScoringNetwork


@dataclass(eq=False)
class AnomalyScoringStep:
    """
    Training loss and per-step anomaly scores of an anomaly scoring network, as the trainer and the detector use them.

    Attributes
    ----------
    network : AnomalyScoringNetwork
        The network that is trained, saved and scores windows.
    model_network : ReconstructionNetwork | ForecastNetwork | AnomalyScoringNetwork
        The network the model built, which ``network`` wraps or is.
    epoch : int
        Zero-based index of the current training epoch, passed to the network's loss.
    """

    network: AnomalyScoringNetwork
    model_network: ReconstructionNetwork | ForecastNetwork | AnomalyScoringNetwork
    epoch: int = field(default=0, init=False)

    def start_epoch(self, epoch: int) -> None:
        """
        Remember the epoch about to start, for losses whose phase weights change during training.

        Parameters
        ----------
        epoch : int
            Zero-based index of the epoch.

        Raises
        ------
        ValueError
            If epoch is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must not be negative, got {epoch}")
        self.epoch = epoch

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the network's own loss of a batch of windows.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled windows.
        is_training : bool
            Unused; the network follows its own training mode.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return self.network.compute_loss(batch[0], self.epoch)

    def compute_scores(self, batch: tuple[Tensor, ...]) -> Tensor:
        """
        Compute the anomaly score of each step of a batch of windows.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled windows with shape (batch, length, channels).

        Returns
        -------
        Tensor
            The scores with shape (batch, length); NaN where the network does not score a step.
        """
        return self.network(batch[0])
