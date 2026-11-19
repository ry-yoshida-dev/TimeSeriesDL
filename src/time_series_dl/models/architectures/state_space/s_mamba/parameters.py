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
from .forecast_network import SMambaForecastNetwork
from .reconstruction_network import SMambaReconstructionNetwork
from .summary_network import SMambaSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class SMambaParameters:
    """
    S-Mamba (Wang et al., 2025): bidirectional Mamba layers over variate tokens, each embedding one channel's series,
    in place of the attention of iTransformer.

    The defaults follow the authors' long-term forecasting scripts.

    Attributes
    ----------
    model_size : int
        Size of each variate token.
    state_size : int
        Size of the state of each channel of the Mamba blocks.
    num_layers : int
        Number of encoder layers.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    dropout : float
        Dropout probability of the embedding, the encoder and the summary head.
    activation : Activation
        Activation of the feed-forward networks and of the summary head.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    model_size: int = 512
    state_size: int = 16
    num_layers: int = 2
    feedforward_size: int = 512
    dropout: float = 0.1
    activation: Activation = Activation.GELU

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("state_size", self.state_size),
            ("num_layers", self.num_layers),
            ("feedforward_size", self.feedforward_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
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
        return SMambaForecastNetwork(self, shape)

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
        return SMambaReconstructionNetwork(self, shape)

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
        return SMambaSummaryNetwork(self, shape)
