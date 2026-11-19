from dataclasses import dataclass

from torch import Tensor

from ...losses import PointLoss
from ...models import ReconstructionNetwork


@dataclass(frozen=True, eq=False)
class ReconstructionStep:
    """
    Point loss of a network reconstructing windows of normal behavior.

    Attributes
    ----------
    network : ReconstructionNetwork
        The network.
    loss : PointLoss
        The loss of each reconstructed value.
    """

    network: ReconstructionNetwork
    loss: PointLoss

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the mean reconstruction loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled windows.
        is_training : bool
            Unused; the loss is the same while training and validating.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return self.loss.compute(self.network(batch[0]), batch[0])

    def compute_errors(self, batch: tuple[Tensor, ...]) -> Tensor:
        """
        Compute the reconstruction loss of each step, averaged over channels.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled windows with shape (batch, length, channels).

        Returns
        -------
        Tensor
            The errors with shape (batch, length).
        """
        return self.loss.compute_elementwise(self.network(batch[0]), batch[0]).mean(dim=2)
