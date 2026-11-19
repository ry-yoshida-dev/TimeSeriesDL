from dataclasses import dataclass

import torch
from torch import Tensor

from ...losses import PointLoss
from ...models.interfaces.masked_reconstruction import MaskedReconstruction
from ...models.interfaces.masked_reconstruction_network import MaskedReconstructionNetwork


@dataclass(frozen=True, eq=False)
class MaskedReconstructionStep:
    """
    Loss of a masked reconstruction network on observed values that were hidden from it, plus the network's own loss.

    Attributes
    ----------
    network : MaskedReconstructionNetwork
        The network.
    mask_ratio : float
        Fraction of the observed values hidden at random while training.
    loss : PointLoss
        The loss of each hidden value.
    """

    network: MaskedReconstructionNetwork
    mask_ratio: float
    loss: PointLoss

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        While training, the loss is the point loss on the hidden values weighted by the network's
        ``hidden_value_loss_weight``, plus the network's own loss; otherwise it is the point loss on the hidden values
        alone, so validation measures imputation in the same way for every network.

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
            The scalar loss.
        """
        values: Tensor = batch[0]
        is_observed: Tensor = batch[1]
        is_hidden: Tensor = (
            (torch.rand_like(values) < self.mask_ratio) & is_observed if is_training else batch[2] & is_observed
        )
        is_input: Tensor = is_observed & ~is_hidden
        output: MaskedReconstruction = self.network(values * is_input, is_input)
        hidden_loss: Tensor = self.loss.compute_masked(output.values, values, is_hidden)
        if not is_training:
            return hidden_loss
        total: Tensor = hidden_loss * self.network.hidden_value_loss_weight
        return total if output.loss is None else total + output.loss

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
        return self.network(batch[0] * batch[1], batch[1]).values
