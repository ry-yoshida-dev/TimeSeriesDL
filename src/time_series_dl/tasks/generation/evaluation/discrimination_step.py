from dataclasses import dataclass

import torch.nn.functional as functional
from torch import Tensor

from .recurrent_scorer import RecurrentScorer


@dataclass(frozen=True, eq=False)
class DiscriminationStep:
    """
    Binary cross-entropy of a post-hoc classifier that tells real (label 1) from synthetic (label 0) sequences by
    its output at the last step.

    Attributes
    ----------
    network : RecurrentScorer
        The classifier, with one output per step.
    """

    network: RecurrentScorer

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The sequences and their labels, 1 for real and 0 for synthetic.
        is_training : bool
            Unused; the loss is the same while training and validating.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return functional.binary_cross_entropy_with_logits(self.network(batch[0])[:, -1, 0], batch[1])
