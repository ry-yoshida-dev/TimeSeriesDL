from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import FourierGNNForecastNetwork


@dataclass(frozen=True, kw_only=True)
class FourierGNNParameters:
    """
    FourierGNN (Yi et al., 2023): every (channel, step) value of the window is a node of one fully connected
    hypervariate graph, and stacked Fourier graph operators perform graph convolutions over it as matrix
    multiplications in the Fourier domain.

    The defaults follow the official code.

    Attributes
    ----------
    embedding_size : int
        Size of the embedding of each value, and of the Fourier graph operators.
    num_operators : int
        Number of stacked Fourier graph operators.
    sparsity_threshold : float
        Threshold of the soft shrinkage applied to the output of every operator.
    time_projection_size : int
        Number of features each channel's steps are projected to before the output network.
    hidden_size : int
        Size of the last hidden layer of the output network.

    Raises
    ------
    ValueError
        If a size is not positive or sparsity_threshold is negative.
    """

    embedding_size: int = 128
    num_operators: int = 3
    sparsity_threshold: float = 0.01
    time_projection_size: int = 8
    hidden_size: int = 256

    def __post_init__(self) -> None:
        for name, value in (
            ("embedding_size", self.embedding_size),
            ("num_operators", self.num_operators),
            ("time_projection_size", self.time_projection_size),
            ("hidden_size", self.hidden_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.sparsity_threshold < 0.0:
            raise ValueError(f"sparsity_threshold must not be negative, got {self.sparsity_threshold}")

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
        return FourierGNNForecastNetwork(self, shape)
