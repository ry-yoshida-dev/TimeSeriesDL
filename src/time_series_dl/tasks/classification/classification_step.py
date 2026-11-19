from dataclasses import dataclass

from torch import Tensor

from ...models import SummaryNetwork
from .classification_loss import ClassificationLoss


@dataclass(frozen=True, eq=False)
class ClassificationStep:
    """
    Classification loss of a summary network's logits on a batch of sequences and class indices.

    Attributes
    ----------
    network : SummaryNetwork
        The network producing one logit per class.
    class_weights : Tensor | None
        Weight of each class in the loss; uniform when None.
    loss : ClassificationLoss
        The loss.
    """

    network: SummaryNetwork
    class_weights: Tensor | None
    loss: ClassificationLoss

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled sequences and their class indices.
        is_training : bool
            Unused; the loss is the same while training and validating.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        logits: Tensor = self.network(batch[0])
        weights: Tensor | None = None if self.class_weights is None else self.class_weights.to(logits.device)
        return self.loss.compute(logits, batch[1], weights)
