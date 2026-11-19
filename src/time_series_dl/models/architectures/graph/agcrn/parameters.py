from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import AGCRNForecastNetwork


@dataclass(frozen=True, kw_only=True)
class AGCRNParameters:
    """
    AGCRN (Bai et al., 2020): gated recurrent units whose linear maps are graph convolutions over a graph generated
    from node embeddings, with node-specific weights generated from the same embeddings.

    The defaults follow the official configuration.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden state of each node.
    num_layers : int
        Number of stacked recurrent layers.
    embedding_size : int
        Size of the node embeddings.
    num_chebyshev_terms : int
        Number of Chebyshev polynomials of the adjacency the graph convolutions propagate with, starting from the
        identity.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    hidden_size: int = 64
    num_layers: int = 2
    embedding_size: int = 10
    num_chebyshev_terms: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("embedding_size", self.embedding_size),
            ("num_chebyshev_terms", self.num_chebyshev_terms),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

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
        return AGCRNForecastNetwork(self, shape)
