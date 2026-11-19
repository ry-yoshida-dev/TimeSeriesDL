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
from .forecast_network import NonStationaryForecastNetwork
from .reconstruction_network import NonStationaryReconstructionNetwork
from .summary_network import NonStationarySummaryNetwork


@dataclass(frozen=True, kw_only=True)
class NonStationaryTransformerParameters:
    """
    Non-stationary Transformer (Liu et al., NeurIPS 2022): series stationarization around a Transformer whose
    de-stationary attention rescales and shifts the scores by factors learned from the raw series' statistics.

    Defaults follow the official code.

    Attributes
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads; it divides model_size.
    num_encoder_layers : int
        Number of encoder layers.
    num_decoder_layers : int
        Number of decoder layers.
    feedforward_size : int
        Hidden size of the position-wise feed-forward networks.
    dropout : float
        Dropout probability of the embeddings, attention weights, sublayer outputs and summary head.
    activation : Activation
        Activation of the feed-forward networks and of the summary head.
    projector_hidden_sizes : tuple[int, ...]
        Hidden layer sizes of the projectors that learn the de-stationary factors; at least one.
    label_length : int
        Number of last past steps that start the decoder input; at most the input length.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, projector_hidden_sizes is empty, or dropout
        is not in [0, 1).
    """

    model_size: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 1
    feedforward_size: int = 2048
    dropout: float = 0.05
    activation: Activation = Activation.GELU
    projector_hidden_sizes: tuple[int, ...] = (128, 128)
    label_length: int = 48

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_encoder_layers", self.num_encoder_layers),
            ("num_decoder_layers", self.num_decoder_layers),
            ("feedforward_size", self.feedforward_size),
            ("label_length", self.label_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads must divide model_size, got {self.num_heads} and {self.model_size}")
        if not self.projector_hidden_sizes:
            raise ValueError("projector_hidden_sizes must not be empty.")
        if any(size <= 0 for size in self.projector_hidden_sizes):
            raise ValueError(f"projector_hidden_sizes must be positive, got {self.projector_hidden_sizes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained encoder-decoder forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network.

        Raises
        ------
        ValueError
            If label_length exceeds the input length.
        """
        return NonStationaryForecastNetwork(self, shape)

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
        return NonStationaryReconstructionNetwork(self, shape)

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
        return NonStationarySummaryNetwork(self, shape)
