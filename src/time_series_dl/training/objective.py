"""Structural type for the task-specific loss computation that the trainer optimizes."""

from typing import Protocol

from torch import Tensor


class TrainingObjective(Protocol):
    """
    Loss of a network on one batch, defined by each task.
    """

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the scalar loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The tensors of one batch, already on the training device.
        is_training : bool
            True during optimization, False during validation; tasks may use training-only behavior such as teacher
            forcing or random masking.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        ...
