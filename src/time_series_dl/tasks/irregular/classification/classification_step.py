from dataclasses import dataclass

from torch import Tensor

from ....models.interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....models.interfaces.irregular_tensors import IrregularTensors
from ...classification import ClassificationLoss


@dataclass(frozen=True, eq=False)
class IrregularClassificationStep:
    """
    Training loss of an irregular summary network on a batch of sequences and class indices: the classification loss
    of its logits, plus whatever the network's paper adds.

    Attributes
    ----------
    network : IrregularSummaryNetwork
        The network producing one logit per class.
    class_weights : Tensor | None
        Weight of each class in the loss; uniform when None.
    loss : ClassificationLoss
        The classification loss.
    """

    network: IrregularSummaryNetwork
    class_weights: Tensor | None
    loss: ClassificationLoss

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled times, values, observed mask and lengths, and the class indices.
        is_training : bool
            Unused; the network samples latent states only in training mode.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        sequences: IrregularTensors = IrregularTensors.from_batch(batch).trimmed()
        indices: Tensor = batch[4]
        weights: Tensor | None = None if self.class_weights is None else self.class_weights.to(indices.device)
        return self.network.compute_training_loss(sequences, lambda logits: self.loss.compute(logits, indices, weights))
