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
from .forecast_network import PatchTSTForecastNetwork
from .reconstruction_network import PatchTSTReconstructionNetwork
from .summary_network import PatchTSTSummaryNetwork
from .token_normalization import TokenNormalization


@dataclass(frozen=True, kw_only=True)
class PatchTSTParameters:
    """
    PatchTST (Nie et al., 2023): channel-independent patches of each series encoded by a Transformer.

    Only supervised training is supported; the masked self-supervised pretraining of the paper is not. The defaults
    are those of the paper's supervised PatchTST/64.

    Attributes
    ----------
    patch_length : int
        Number of time steps of each patch.
    stride : int
        Number of time steps between the starts of consecutive patches; also the length of the end padding.
    model_size : int
        Size of each patch token.
    num_heads : int
        Number of attention heads; must divide model_size.
    num_layers : int
        Number of Transformer encoder layers.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    dropout : float
        Dropout probability inside the encoder.
    head_dropout : float
        Dropout probability before the output layer.
    activation : Activation
        Activation of the feed-forward networks.
    normalization : TokenNormalization
        Normalization of the tokens inside the encoder.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or a dropout is not in [0, 1).
    """

    patch_length: int = 16
    stride: int = 8
    model_size: int = 128
    num_heads: int = 16
    num_layers: int = 3
    feedforward_size: int = 256
    dropout: float = 0.2
    head_dropout: float = 0.0
    activation: Activation = Activation.GELU
    normalization: TokenNormalization = TokenNormalization.BATCH

    def __post_init__(self) -> None:
        for name, value in (
            ("patch_length", self.patch_length),
            ("stride", self.stride),
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("feedforward_size", self.feedforward_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must divide model_size {self.model_size}")
        for name, probability in (("dropout", self.dropout), ("head_dropout", self.head_dropout)):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {probability}")

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
        return PatchTSTForecastNetwork(self, shape)

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
        return PatchTSTReconstructionNetwork(self, shape)

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
        return PatchTSTSummaryNetwork(self, shape)
