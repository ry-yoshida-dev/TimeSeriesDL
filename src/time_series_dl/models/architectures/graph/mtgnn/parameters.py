from dataclasses import dataclass

from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from .forecast_network import MTGNNForecastNetwork
from .reconstruction_network import MTGNNReconstructionNetwork
from .summary_network import MTGNNSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class MTGNNParameters:
    """
    MTGNN (Wu et al., 2020): a graph learning layer infers a sparse uni-directed graph over the channels, and layers
    of gated dilated inception convolutions along time and mix-hop propagation over the graph feed skip connections
    into a pointwise output module.

    The defaults follow the official multi-step forecasting script.

    Attributes
    ----------
    num_layers : int
        Number of temporal and graph convolution layers.
    convolution_channels : int
        Output channels of the dilated inception convolutions; must be divisible by 4.
    residual_channels : int
        Channels of the residual stream.
    skip_channels : int
        Channels of the skip connections.
    end_channels : int
        Channels of the hidden layer of the output module.
    embedding_size : int
        Size of the node embeddings of the graph learning layer.
    num_neighbors : int
        Number of neighbors each node keeps in the learned graph.
    saturation : float
        Rate of the hyperbolic tangents of the graph learning layer.
    propagation_depth : int
        Number of propagation steps of the mix-hop propagation; at least one, or the graph would be unused.
    retain_ratio : float
        Share of the input that mix-hop propagation keeps at every step.
    dilation_base : int
        Factor by which the dilation grows from layer to layer.
    dropout : float
        Dropout probability of the input skip connection, of the gated convolutions and of the summary head.

    Raises
    ------
    ValueError
        If a size or the saturation is not positive, convolution_channels is not divisible by 4, retain_ratio is not
        in [0, 1], or dropout is not in [0, 1).
    """

    num_layers: int = 3
    convolution_channels: int = 32
    residual_channels: int = 32
    skip_channels: int = 64
    end_channels: int = 128
    embedding_size: int = 40
    num_neighbors: int = 20
    saturation: float = 3.0
    propagation_depth: int = 2
    retain_ratio: float = 0.05
    dilation_base: int = 1
    dropout: float = 0.3

    def __post_init__(self) -> None:
        for name, value in (
            ("num_layers", self.num_layers),
            ("convolution_channels", self.convolution_channels),
            ("residual_channels", self.residual_channels),
            ("skip_channels", self.skip_channels),
            ("end_channels", self.end_channels),
            ("embedding_size", self.embedding_size),
            ("num_neighbors", self.num_neighbors),
            ("dilation_base", self.dilation_base),
            ("propagation_depth", self.propagation_depth),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.convolution_channels % 4 != 0:
            raise ValueError(f"convolution_channels must be divisible by 4, got {self.convolution_channels}")
        if self.saturation <= 0.0:
            raise ValueError(f"saturation must be positive, got {self.saturation}")
        if not 0.0 <= self.retain_ratio <= 1.0:
            raise ValueError(f"retain_ratio must be in [0, 1], got {self.retain_ratio}")
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
        return MTGNNForecastNetwork(self, shape)

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
        return MTGNNReconstructionNetwork(self, shape)

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
        return MTGNNSummaryNetwork(self, shape)
