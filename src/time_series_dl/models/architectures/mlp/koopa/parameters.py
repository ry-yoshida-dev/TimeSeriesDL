from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import KoopaForecastNetwork


@dataclass(frozen=True, kw_only=True)
class KoopaParameters:
    """
    Koopa (Liu et al., 2023): stacked blocks of a Fourier filter, a time-invariant Koopman predictor with a learned
    global operator and a time-variant Koopman predictor with an operator fitted to each window.

    The defaults follow Time-Series-Library, whose two "hidden layers" are the two linear layers of each MLP.

    Attributes
    ----------
    num_blocks : int
        Number of stacked blocks.
    dynamic_size : int
        Size of the Koopman states.
    hidden_size : int
        Size of the hidden layers of the encoder and decoder MLPs.
    num_layers : int
        Number of linear layers of each encoder and decoder MLP.
    invariant_fraction : float
        Fraction of the frequencies, those of largest amplitude, that form the time-invariant part.
    dropout : float
        Dropout probability after each hidden layer of the MLPs.

    Raises
    ------
    ValueError
        If a size or count is not positive, invariant_fraction is not in [0, 1] or dropout is not in [0, 1).
    """

    num_blocks: int = 3
    dynamic_size: int = 128
    hidden_size: int = 64
    num_layers: int = 2
    invariant_fraction: float = 0.2
    dropout: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("dynamic_size", self.dynamic_size),
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.invariant_fraction <= 1.0:
            raise ValueError(f"invariant_fraction must be in [0, 1], got {self.invariant_fraction}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; the input must be longer than the horizon.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return KoopaForecastNetwork(self, shape)
