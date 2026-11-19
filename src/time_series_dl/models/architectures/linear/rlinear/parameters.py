from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape, ReconstructionNetwork, SequenceShape
from .forecast_network import RLinearForecastNetwork
from .reconstruction_network import RLinearReconstructionNetwork


@dataclass(frozen=True, kw_only=True)
class RLinearParameters:
    """
    RLinear (Li et al., 2023): one linear map along time between a reversible instance normalization with a learnable
    affine transformation and its inverse.

    The defaults follow the official code (shared weights, dropout 0.1, affine RevIN).

    Attributes
    ----------
    is_individual : bool
        Whether each channel has its own linear map.
    dropout : float
        Dropout probability on the normalized window.
    is_affine : bool
        Whether the instance normalization has a learnable per-channel affine transformation.

    Raises
    ------
    ValueError
        If dropout is not in [0, 1).
    """

    is_individual: bool = False
    dropout: float = 0.1
    is_affine: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return RLinearForecastNetwork(self, shape)

    def build_reconstruction_network(self, shape: SequenceShape) -> ReconstructionNetwork:
        """
        Build an untrained reconstruction network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences.

        Returns
        -------
        ReconstructionNetwork
            The network.
        """
        return RLinearReconstructionNetwork(self, shape)
