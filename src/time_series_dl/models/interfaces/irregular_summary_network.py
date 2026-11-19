from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

from torch import Tensor, nn

from .irregular_tensors import IrregularTensors


class IrregularSummaryNetwork(nn.Module, ABC):
    """
    Maps irregularly sampled sequences to one output vector each, such as class logits or regression targets.

    Networks whose papers train with more than the task's loss, such as the evidence lower bound of a variational
    model, add it in ``compute_training_loss``; the task supplies only its loss of the outputs.
    """

    @abstractmethod
    def forward(self, sequences: IrregularTensors) -> Tensor:
        """
        Summarize a batch of sequences.

        Parameters
        ----------
        sequences : IrregularTensors
            The padded sequences.

        Returns
        -------
        Tensor
            Outputs with shape (batch, output_size), unchanged by padding steps.
        """

    def compute_training_loss(self, sequences: IrregularTensors, output_loss: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Compute the loss that training minimizes; by default the task's loss of the outputs.

        Parameters
        ----------
        sequences : IrregularTensors
            The padded sequences.
        output_loss : Callable[[Tensor], Tensor]
            The task's scalar loss of outputs with shape (batch, output_size).

        Returns
        -------
        Tensor
            The scalar loss.
        """
        return output_loss(self(sequences))

    def __call__(self, sequences: IrregularTensors) -> Tensor:
        return cast(Tensor, super().__call__(sequences))
