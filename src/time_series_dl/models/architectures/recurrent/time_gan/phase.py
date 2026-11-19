from enum import Enum


class TimeGANPhase(Enum):
    """
    Training phase of TimeGAN.

    Attributes
    ----------
    AUTOENCODER
        The embedder and the recovery network learn to reconstruct the sequences.
    SUPERVISED
        The supervisor learns to predict the next latent step of real sequences.
    JOINT
        The generator, supervisor, embedder, recovery network and discriminator are trained together, adversarially.
    """

    AUTOENCODER = "autoencoder"
    SUPERVISED = "supervised"
    JOINT = "joint"
