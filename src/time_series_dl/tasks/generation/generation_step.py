from dataclasses import dataclass, field

from torch import Tensor

from ...models.interfaces.sequence_generation_network import SequenceGenerationNetwork


@dataclass(eq=False)
class GenerationStep:
    """
    Training loss of a sequence generation network, as the trainer uses it, with the epoch for networks trained in
    phases.

    Attributes
    ----------
    network : SequenceGenerationNetwork
        The network that is trained, saved and samples.
    epoch : int
        Zero-based index of the current training epoch, passed to the network's loss.
    """

    network: SequenceGenerationNetwork
    epoch: int = field(default=0, init=False)

    def start_epoch(self, epoch: int) -> None:
        """
        Remember the epoch about to start, for losses whose phases change during training.

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
        Compute the network's own loss of a batch of sequences.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled sequences.
        is_training : bool
            Unused; the network follows its own training mode.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return self.network.compute_loss(batch[0], self.epoch)
