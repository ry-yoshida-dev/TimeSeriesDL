"""Structural type for training objectives whose loss changes with the epoch."""

from typing import Protocol, runtime_checkable

from .objective import TrainingObjective


@runtime_checkable
class EpochScheduledObjective(TrainingObjective, Protocol):
    """
    Loss of a network on one batch that depends on how far training has progressed, such as the phase weights of
    adversarially trained anomaly detectors. The trainer announces each epoch before its first batch.
    """

    def start_epoch(self, epoch: int) -> None:
        """
        Prepare the losses of an epoch, for training and for validation alike.

        Parameters
        ----------
        epoch : int
            Zero-based index of the epoch about to start.
        """
        ...
