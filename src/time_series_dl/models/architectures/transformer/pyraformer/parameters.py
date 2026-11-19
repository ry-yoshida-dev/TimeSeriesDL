from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import PyraformerForecastNetwork
from .reconstruction_network import PyraformerReconstructionNetwork
from .summary_network import PyraformerSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class PyraformerParameters:
    """
    Pyraformer (Liu et al., ICLR 2022): attention over a pyramidal graph of the series at several resolutions, built by
    strided convolutions, so that information passes between any two steps in a constant number of layers.

    The defaults follow the Time-Series-Library, whose bottleneck size is a quarter of the model size.

    Attributes
    ----------
    window_sizes : tuple[int, ...]
        Number of children of each node of every coarser scale, from the finest; one coarser scale per entry.
    inner_size : int
        Number of neighbours, centred on each node and including itself, that it attends to within its scale.
    model_size : int
        Size of each node token.
    num_heads : int
        Number of attention heads; must divide model_size.
    num_layers : int
        Number of pyramidal attention layers.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    bottleneck_size : int
        Size of the tokens inside the coarser-scale construction convolutions.
    dropout : float
        Dropout probability of the embedding, attention weights, sublayer outputs and summary head.

    Raises
    ------
    ValueError
        If a size is not positive, window_sizes is empty or has a size below 2, num_heads does not divide model_size,
        or dropout is not in [0, 1).
    """

    window_sizes: tuple[int, ...] = (4, 4)
    inner_size: int = 5
    model_size: int = 512
    num_heads: int = 8
    num_layers: int = 2
    feedforward_size: int = 2048
    bottleneck_size: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("inner_size", self.inner_size),
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("feedforward_size", self.feedforward_size),
            ("bottleneck_size", self.bottleneck_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not self.window_sizes or any(size < 2 for size in self.window_sizes):
            raise ValueError(f"window_sizes must be non-empty sizes of at least 2, got {self.window_sizes}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must divide model_size {self.model_size}")
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
        return PyraformerForecastNetwork(self, shape)

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
        return PyraformerReconstructionNetwork(self, shape)

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
        return PyraformerSummaryNetwork(self, shape)
