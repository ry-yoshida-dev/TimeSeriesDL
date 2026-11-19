from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingHistory:
    """
    Losses of every epoch of one training run.

    Attributes
    ----------
    training_losses : tuple[float, ...]
        Mean training loss of each epoch.
    validation_losses : tuple[float, ...]
        Validation loss of each epoch; empty when no validation data was given.
    best_epoch : int
        Zero-based epoch whose weights were kept.
    is_stopped_early : bool
        Whether training stopped before the maximum number of epochs.
    """

    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    is_stopped_early: bool

    @property
    def num_epochs(self) -> int:
        """
        Get the number of completed epochs.

        Returns
        -------
        int
            The number of epochs.
        """
        return len(self.training_losses)

    @property
    def best_loss(self) -> float:
        """
        Get the monitored loss of the kept epoch.

        Returns
        -------
        float
            The validation loss of the best epoch, or its training loss without validation data.
        """
        losses: tuple[float, ...] = self.validation_losses if self.validation_losses else self.training_losses
        return losses[self.best_epoch]
