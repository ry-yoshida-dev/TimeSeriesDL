from dataclasses import dataclass

from ....components.layers import Activation
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import MLPForecastNetwork
from .reconstruction_network import MLPReconstructionNetwork
from .summary_network import MLPSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class MLPParameters:
    """
    Multilayer perceptron over the flattened window, the simplest baseline of every task.

    Attributes
    ----------
    hidden_sizes : tuple[int, ...]
        Size of each hidden layer.
    activation : Activation
        Activation after each hidden layer.
    dropout : float
        Dropout probability after each hidden activation.

    Raises
    ------
    ValueError
        If a hidden size is not positive or dropout is not in [0, 1).
    """

    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: Activation = Activation.RELU
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive, got {self.hidden_sizes}")
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
        return MLPForecastNetwork(self, shape)

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
        return MLPReconstructionNetwork(self, shape)

    def build_summary_network(self, shape: SummaryShape) -> SummaryNetwork:
        """
        Build an untrained summary network.

        Parameters
        ----------
        shape : SummaryShape
            The input and output dimensions.

        Returns
        -------
        SummaryNetwork
            The network.
        """
        return MLPSummaryNetwork(self, shape)
