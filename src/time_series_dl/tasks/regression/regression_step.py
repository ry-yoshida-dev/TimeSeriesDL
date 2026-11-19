from dataclasses import dataclass

from deep_learning_loss import PointLoss
from torch import Tensor

from ...models import SummaryNetwork


@dataclass(frozen=True, eq=False)
class RegressionStep:
    """
    Point loss of a summary network's outputs on a batch of sequences and scaled targets.

    Attributes
    ----------
    network : SummaryNetwork
        The network producing one value per target.
    loss : PointLoss
        The loss.
    """

    network: SummaryNetwork
    loss: PointLoss

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled sequences and their scaled targets.
        is_training : bool
            Unused; the loss is the same while training and validating.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return self.loss.compute(self.network(batch[0]), batch[1])
