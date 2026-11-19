from enum import Enum

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LRScheduler, OneCycleLR


class LearningRateSchedule(Enum):
    """
    Schedule of the learning rate over the epochs.

    Attributes
    ----------
    CONSTANT
        The initial learning rate throughout.
    COSINE
        Cosine annealing from the initial learning rate to zero at the last epoch.
    ONE_CYCLE
        Warm-up to the initial learning rate, then annealing, stepped per batch.
    """

    CONSTANT = "constant"
    COSINE = "cosine"
    ONE_CYCLE = "one_cycle"

    @property
    def is_stepped_per_batch(self) -> bool:
        """
        Check whether the scheduler steps after every batch instead of every epoch.

        Returns
        -------
        bool
            True for schedules defined over optimizer steps.
        """
        match self:
            case LearningRateSchedule.ONE_CYCLE:
                return True
            case LearningRateSchedule.CONSTANT | LearningRateSchedule.COSINE:
                return False

    def build(
        self, optimizer: torch.optim.Optimizer, learning_rate: float, max_epochs: int, steps_per_epoch: int
    ) -> LRScheduler:
        """
        Build the scheduler of an optimizer.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate is scheduled.
        learning_rate : float
            The initial (peak) learning rate.
        max_epochs : int
            The number of epochs the schedule spans.
        steps_per_epoch : int
            The number of batches per epoch.

        Returns
        -------
        LRScheduler
            The scheduler.
        """
        match self:
            case LearningRateSchedule.CONSTANT:
                return ConstantLR(optimizer, factor=1.0, total_iters=0)
            case LearningRateSchedule.COSINE:
                return CosineAnnealingLR(optimizer, T_max=max_epochs)
            case LearningRateSchedule.ONE_CYCLE:
                return OneCycleLR(optimizer, max_lr=learning_rate, epochs=max_epochs, steps_per_epoch=steps_per_epoch)
