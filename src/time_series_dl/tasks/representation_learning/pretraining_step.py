from dataclasses import dataclass

from torch import Tensor

from ...models.interfaces.pretraining_network import PretrainingNetwork


@dataclass(frozen=True, eq=False)
class PretrainingStep:
    """
    Self-supervised loss of a pretraining network on a batch of unlabeled sequences.

    Attributes
    ----------
    network : PretrainingNetwork
        The network, which draws its own augmentations, masks or crops.
    """

    network: PretrainingNetwork

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled sequences.
        is_training : bool
            Unused; validation losses also draw new random views, so they vary between epochs.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return self.network(batch[0])
