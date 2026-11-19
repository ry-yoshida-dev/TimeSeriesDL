from dataclasses import dataclass

import torch
from torch import Tensor

from ...losses import PointLoss
from ...models import ReconstructionNetwork


@dataclass(frozen=True, eq=False)
class MaskedReconstructionStep:
    """
    Point loss of a reconstruction network on observed values that were hidden from it.

    Attributes
    ----------
    network : ReconstructionNetwork
        The network.
    mask_ratio : float
        Fraction of the observed values hidden at random while training.
    loss : PointLoss
        The loss of each hidden value.
    """

    network: ReconstructionNetwork
    mask_ratio: float
    loss: PointLoss

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled values with missing values set to zero, the observed mask, and a fixed mask of the values to
            hide outside training.
        is_training : bool
            Whether a new random set of values is hidden instead of the fixed one.

        Returns
        -------
        Tensor
            The mean loss over the hidden values.
        """
        values: Tensor = batch[0]
        is_observed: Tensor = batch[1]
        is_hidden: Tensor = (
            (torch.rand_like(values) < self.mask_ratio) & is_observed if is_training else batch[2] & is_observed
        )
        reconstructed: Tensor = self.network(values * (is_observed & ~is_hidden))
        return self.loss.compute_masked(reconstructed, values, is_hidden)

    def reconstruct(self, batch: tuple[Tensor, ...]) -> Tensor:
        """
        Reconstruct a batch from its observed values.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled values with missing values set to zero, and the observed mask.

        Returns
        -------
        Tensor
            The reconstruction of every value.
        """
        return self.network(batch[0] * batch[1])
