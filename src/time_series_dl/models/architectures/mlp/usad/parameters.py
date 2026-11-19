from dataclasses import dataclass

from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from .anomaly_scoring_network import USADAnomalyScoringNetwork


@dataclass(frozen=True, kw_only=True)
class USADParameters:
    """
    USAD (Audibert et al., 2020): two autoencoders sharing one encoder over the flattened window, trained
    adversarially, scoring each step by a weighted sum of its two reconstruction errors.

    The defaults follow the official implementation. The reconstructions end with a sigmoid, so the detector should
    scale the channels to [0, 1] (``MinMaxScalingParameters``), as in the paper.

    Attributes
    ----------
    latent_size_per_step : int
        Size of the latent code per step of the window; the code has ``window_length * latent_size_per_step``
        values.
    alpha : float
        Weight of the first autoencoder's reconstruction error in the anomaly score.
    beta : float
        Weight of the second autoencoder's reconstruction of the first one's output in the anomaly score.

    Raises
    ------
    ValueError
        If latent_size_per_step is not positive, a weight is negative, or both weights are zero.
    """

    latent_size_per_step: int = 100
    alpha: float = 0.5
    beta: float = 0.5

    def __post_init__(self) -> None:
        if self.latent_size_per_step <= 0:
            raise ValueError(f"latent_size_per_step must be positive, got {self.latent_size_per_step}")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError(f"alpha and beta must not be negative, got {self.alpha} and {self.beta}")
        if self.alpha + self.beta == 0.0:
            raise ValueError("alpha and beta must not both be zero.")

    def build_anomaly_scoring_network(self, shape: SequenceShape) -> AnomalyScoringNetwork:
        """
        Build an untrained anomaly scoring network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the windows.

        Returns
        -------
        AnomalyScoringNetwork
            The network.
        """
        return USADAnomalyScoringNetwork(self, shape)
