from abc import ABC, abstractmethod

from torch import Tensor


class ClassificationLoss(ABC):
    """
    Loss of class logits given the true class indices.
    """

    @abstractmethod
    def compute(self, logits: Tensor, indices: Tensor, class_weights: Tensor | None) -> Tensor:
        """
        Compute the mean loss.

        Parameters
        ----------
        logits : Tensor
            Unnormalized scores with shape (batch, classes).
        indices : Tensor
            True class indices with shape (batch,).
        class_weights : Tensor | None
            Weight of each class with shape (classes,); the mean is then weighted by the weight of each true class.
            Uniform when None.

        Returns
        -------
        Tensor
            The scalar loss.
        """
