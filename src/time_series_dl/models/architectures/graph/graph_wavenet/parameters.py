from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import GraphWaveNetForecastNetwork


@dataclass(frozen=True, kw_only=True)
class GraphWaveNetParameters:
    """
    Graph WaveNet (Wu et al., 2019): a self-adaptive adjacency learned from node embeddings, and blocks of gated
    dilated temporal convolutions and diffusion graph convolutions whose skip connections feed the output module.

    The defaults follow the official training script.

    Attributes
    ----------
    residual_channels : int
        Channels of the residual stream.
    dilation_channels : int
        Output channels of the gated temporal convolutions.
    skip_channels : int
        Channels of the skip connections.
    end_channels : int
        Channels of the hidden layer of the output module.
    kernel_size : int
        Length of the temporal convolution kernels.
    num_blocks : int
        Number of blocks, each restarting the dilation at one.
    num_layers : int
        Number of layers per block; the dilation doubles from layer to layer.
    embedding_size : int
        Size of the node embeddings of the adaptive adjacency.
    diffusion_order : int
        Highest power of the transition matrix in the diffusion convolutions.
    dropout : float
        Dropout probability of the diffusion convolutions.

    Raises
    ------
    ValueError
        If a size is not positive, kernel_size is below 2, or dropout is not in [0, 1).
    """

    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 256
    end_channels: int = 512
    kernel_size: int = 2
    num_blocks: int = 4
    num_layers: int = 2
    embedding_size: int = 10
    diffusion_order: int = 2
    dropout: float = 0.3

    def __post_init__(self) -> None:
        for name, value in (
            ("residual_channels", self.residual_channels),
            ("dilation_channels", self.dilation_channels),
            ("skip_channels", self.skip_channels),
            ("end_channels", self.end_channels),
            ("num_blocks", self.num_blocks),
            ("num_layers", self.num_layers),
            ("embedding_size", self.embedding_size),
            ("diffusion_order", self.diffusion_order),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.kernel_size < 2:
            raise ValueError(f"kernel_size must be at least 2, got {self.kernel_size}")
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
        return GraphWaveNetForecastNetwork(self, shape)
