from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import StemGNNForecastNetwork


@dataclass(frozen=True, kw_only=True)
class StemGNNParameters:
    """
    StemGNN (Cao et al., 2020): Spectral Temporal Graph Neural Network, which infers a graph from the window by
    self-attention and models each window in the graph spectral domain (graph Fourier filters) and the frequency
    domain (discrete Fourier transform) jointly.

    The defaults follow the official code, except latent_size, which the official code ties to the number of nodes.

    Attributes
    ----------
    num_blocks : int
        Number of StemGNN blocks; each block but the last passes a backcast to the next.
    expansion_factor : int
        Factor by which the spectral sequential cells lengthen each sequence.
    num_chebyshev_terms : int
        Number of Chebyshev polynomials of the Laplacian used as graph Fourier filters.
    latent_size : int
        Hidden size of the recurrent unit of the latent correlation layer.
    attention_negative_slope : float
        Negative slope of the leaky rectifier of the self-attention scores.
    dropout : float
        Dropout probability of the attention.

    Raises
    ------
    ValueError
        If a size is not positive, attention_negative_slope is negative, or dropout is not in [0, 1).
    """

    num_blocks: int = 2
    expansion_factor: int = 5
    num_chebyshev_terms: int = 4
    latent_size: int = 64
    attention_negative_slope: float = 0.2
    dropout: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("expansion_factor", self.expansion_factor),
            ("num_chebyshev_terms", self.num_chebyshev_terms),
            ("latent_size", self.latent_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.attention_negative_slope < 0.0:
            raise ValueError(f"attention_negative_slope must not be negative, got {self.attention_negative_slope}")
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
        return StemGNNForecastNetwork(self, shape)
