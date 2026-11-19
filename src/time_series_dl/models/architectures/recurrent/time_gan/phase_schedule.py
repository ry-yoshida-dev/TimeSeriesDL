from dataclasses import dataclass

from .phase import TimeGANPhase


@dataclass(frozen=True, kw_only=True)
class TimeGANPhaseSchedule:
    """
    Epochs of the three training phases of TimeGAN.

    The autoencoder phase takes the first ``autoencoder_epochs`` epochs, the supervised phase the next
    ``supervised_epochs`` and the joint phase every epoch after them, up to the trainer's ``max_epochs``.

    Attributes
    ----------
    autoencoder_epochs : int
        Number of epochs of the autoencoder phase.
    supervised_epochs : int
        Number of epochs of the supervised phase.

    Raises
    ------
    ValueError
        If a number of epochs is negative.
    """

    autoencoder_epochs: int = 33
    supervised_epochs: int = 33

    def __post_init__(self) -> None:
        if self.autoencoder_epochs < 0:
            raise ValueError(f"autoencoder_epochs must not be negative, got {self.autoencoder_epochs}")
        if self.supervised_epochs < 0:
            raise ValueError(f"supervised_epochs must not be negative, got {self.supervised_epochs}")

    @property
    def first_joint_epoch(self) -> int:
        """
        Get the epoch the joint phase starts at.

        Returns
        -------
        int
            The zero-based epoch.
        """
        return self.autoencoder_epochs + self.supervised_epochs

    def phase_at(self, epoch: int) -> TimeGANPhase:
        """
        Get the phase of an epoch.

        Parameters
        ----------
        epoch : int
            Zero-based index of the epoch.

        Returns
        -------
        TimeGANPhase
            The phase.

        Raises
        ------
        ValueError
            If epoch is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must not be negative, got {epoch}")
        if epoch < self.autoencoder_epochs:
            return TimeGANPhase.AUTOENCODER
        if epoch < self.first_joint_epoch:
            return TimeGANPhase.SUPERVISED
        return TimeGANPhase.JOINT

    def epochs_until_joint(self, epoch: int) -> int:
        """
        Get the number of epochs left before the joint phase.

        Parameters
        ----------
        epoch : int
            Zero-based index of the epoch.

        Returns
        -------
        int
            The number of epochs from this one to the first joint epoch; 0 within the joint phase.

        Raises
        ------
        ValueError
            If epoch is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must not be negative, got {epoch}")
        return max(self.first_joint_epoch - epoch, 0)
