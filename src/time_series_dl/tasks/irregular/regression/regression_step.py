from dataclasses import dataclass

from deep_learning_loss import PointLoss
from torch import Tensor

from ....models.interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....models.interfaces.irregular_tensors import IrregularTensors


@dataclass(frozen=True, eq=False)
class IrregularRegressionStep:
    """
    Training loss of an irregular summary network on a batch of sequences and scaled targets: the point loss of its
    outputs, plus whatever the network's paper adds.

    Attributes
    ----------
    network : IrregularSummaryNetwork
        The network producing one value per target.
    loss : PointLoss
        The point loss.
    """

    network: IrregularSummaryNetwork
    loss: PointLoss

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled times, values, observed mask and lengths, and the scaled targets.
        is_training : bool
            Unused; the network samples latent states only in training mode.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        targets: Tensor = batch[4]
        return self.network.compute_training_loss(
            IrregularTensors.from_batch(batch).trimmed(), lambda outputs: self.loss.compute(outputs, targets)
        )
