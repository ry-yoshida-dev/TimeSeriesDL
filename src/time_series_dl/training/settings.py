from dataclasses import dataclass

from ..runtime import Device
from .learning_rate_schedule import LearningRateSchedule
from .optimizer_kind import OptimizerKind


@dataclass(frozen=True, kw_only=True)
class TrainingSettings:
    """
    How a network is trained, identical for every model and task.

    Attributes
    ----------
    max_epochs : int
        Maximum number of passes over the training data.
    batch_size : int
        Number of samples per optimizer step.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        Weight decay coefficient.
    optimizer : OptimizerKind
        Optimizer of the network parameters.
    schedule : LearningRateSchedule
        Schedule of the learning rate.
    early_stopping_patience : int | None
        Epochs without improvement of the monitored loss before training stops; no early stopping when None. The
        validation loss is monitored when validation data is given, the training loss otherwise. The weights of the
        best epoch are restored in either case.
    gradient_clip_norm : float | None
        Maximum norm of the gradients; no clipping when None.
    device : Device
        Device to train and run on.
    random_seed : int | None
        Seed of weight initialization, shuffling and masking; not seeded when None.
    is_verbose : bool
        Whether the loss of every epoch is printed.

    Raises
    ------
    ValueError
        If a count, rate or norm is not positive, or weight_decay is negative.
    """

    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerKind = OptimizerKind.ADAMW
    schedule: LearningRateSchedule = LearningRateSchedule.CONSTANT
    early_stopping_patience: int | None = 10
    gradient_clip_norm: float | None = None
    device: Device = Device.AUTO
    random_seed: int | None = 0
    is_verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must not be negative, got {self.weight_decay}")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive when set, got {self.early_stopping_patience}")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError(f"gradient_clip_norm must be positive when set, got {self.gradient_clip_norm}")
