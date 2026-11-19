from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn

from .irregular_tensors import IrregularTensors


class IrregularInterpolationNetwork(nn.Module, ABC):
    """
    Predicts the values of irregularly sampled sequences at any query times from their observed values, inside the
    observed time range (interpolation) or after it (forecasting).

    Networks whose papers train with another objective than the squared error of the observed targets, such as the
    evidence lower bound of a variational model, override ``compute_training_loss``.
    """

    @abstractmethod
    def forward(self, context: IrregularTensors, query_times: Tensor) -> Tensor:
        """
        Predict the values at the query times.

        Parameters
        ----------
        context : IrregularTensors
            The observed values that the predictions are conditioned on.
        query_times : Tensor
            Non-decreasing times of each sample with shape (batch, num_queries); padding repeats the last time.

        Returns
        -------
        Tensor
            The predicted values (the mean of a probabilistic prediction) with shape (batch, num_queries, channels).
        """

    def compute_training_loss(self, context: IrregularTensors, targets: IrregularTensors) -> Tensor:
        """
        Compute the loss that training minimizes; by default the mean squared error of the observed targets.

        Parameters
        ----------
        context : IrregularTensors
            The observed values that the predictions are conditioned on.
        targets : IrregularTensors
            The values to predict, at their own times; only observed values count.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        predictions: Tensor = self(context, targets.times)
        squared_errors: Tensor = (predictions - targets.values).square() * targets.observed_mask
        return squared_errors.sum() / targets.observed_mask.sum().clamp(min=1.0)

    def __call__(self, context: IrregularTensors, query_times: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(context, query_times))
