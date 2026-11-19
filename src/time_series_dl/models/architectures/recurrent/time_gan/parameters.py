from dataclasses import dataclass, field

from ....interfaces import SequenceShape
from ....interfaces.sequence_generation_network import SequenceGenerationNetwork
from ..vanilla import RecurrentCell
from .phase_schedule import TimeGANPhaseSchedule
from .sequence_generation_network import TimeGANSequenceGenerationNetwork


@dataclass(frozen=True, kw_only=True)
class TimeGANParameters:
    """
    TimeGAN (Yoon et al., 2019): a recurrent autoencoder whose latent space a recurrent generator learns with an
    adversarial loss on latent sequences and a supervised loss on their next latent steps.

    The defaults are those of the authors' implementation: GRU networks of 3 layers with 24 hidden units (2 layers in
    the supervisor), noise of the size of the channels, gamma 1 and a discriminator trained only while its loss exceeds
    0.15. The paper trains each phase for the same number of iterations, so the default schedule gives each phase a
    third of the default 100 training epochs; ``TrainingSettings.max_epochs`` must leave at least one joint epoch.

    Attributes
    ----------
    cell : RecurrentCell
        Recurrent cell of all five networks.
    hidden_size : int
        Size of the hidden states and of the latent vectors.
    num_layers : int
        Number of stacked recurrent layers of each network; the supervisor has one less (at least one).
    noise_size : int | None
        Size of the generator's uniform noise vector of each step; the number of channels when None.
    schedule : TimeGANPhaseSchedule
        Epochs of the autoencoder and supervised phases; the joint phase takes the remaining epochs.
    embedding_adversarial_weight : float
        Weight (gamma) of the adversarial terms on the generator's outputs before the supervisor.
    discriminator_loss_threshold : float
        The discriminator is trained on a batch only while its loss exceeds this value.

    Raises
    ------
    ValueError
        If a size or count is not positive, or a weight or the threshold is negative.
    """

    cell: RecurrentCell = RecurrentCell.GRU
    hidden_size: int = 24
    num_layers: int = 3
    noise_size: int | None = None
    schedule: TimeGANPhaseSchedule = field(default_factory=TimeGANPhaseSchedule)
    embedding_adversarial_weight: float = 1.0
    discriminator_loss_threshold: float = 0.15

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.noise_size is not None and self.noise_size <= 0:
            raise ValueError(f"noise_size must be positive when set, got {self.noise_size}")
        if self.embedding_adversarial_weight < 0.0:
            raise ValueError(
                f"embedding_adversarial_weight must not be negative, got {self.embedding_adversarial_weight}"
            )
        if self.discriminator_loss_threshold < 0.0:
            raise ValueError(
                f"discriminator_loss_threshold must not be negative, got {self.discriminator_loss_threshold}"
            )

    def build_sequence_generation_network(self, shape: SequenceShape) -> SequenceGenerationNetwork:
        """
        Build an untrained TimeGAN network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences it generates; at least two steps long.

        Returns
        -------
        SequenceGenerationNetwork
            The network.

        Raises
        ------
        ValueError
            If the sequences are shorter than two steps.
        """
        return TimeGANSequenceGenerationNetwork(self, shape)
