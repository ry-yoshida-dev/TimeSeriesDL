from abc import ABC, abstractmethod

from torch import Tensor


class PointLoss(ABC):
    """
    Loss between point predictions and targets, defined by the error of each value.
    """

    @abstractmethod
    def compute_elementwise(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the loss of each value.

        Parameters
        ----------
        predictions : Tensor
            The predictions.
        targets : Tensor
            The targets, with the shape of the predictions.

        Returns
        -------
        Tensor
            The non-negative loss of each value, with the shape of the predictions.
        """

    def compute(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the mean loss.

        Parameters
        ----------
        predictions : Tensor
            The predictions.
        targets : Tensor
            The targets, with the shape of the predictions.

        Returns
        -------
        Tensor
            The scalar mean loss.
        """
        return self.compute_elementwise(predictions, targets).mean()

    def compute_masked(self, predictions: Tensor, targets: Tensor, is_included: Tensor) -> Tensor:
        """
        Compute the mean loss over the values a mask selects.

        Parameters
        ----------
        predictions : Tensor
            The predictions.
        targets : Tensor
            The targets, with the shape of the predictions.
        is_included : Tensor
            Boolean mask of the shape of the predictions marking the values to average.

        Returns
        -------
        Tensor
            The scalar mean loss; zero when the mask selects nothing.
        """
        losses: Tensor = self.compute_elementwise(predictions, targets) * is_included
        return losses.sum() / is_included.sum().clamp(min=1)
