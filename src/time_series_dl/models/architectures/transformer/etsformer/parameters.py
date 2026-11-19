from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape, SummaryNetwork, SummaryShape
from .forecast_network import ETSformerForecastNetwork
from .summary_network import ETSformerSummaryNetwork


@dataclass(frozen=True, kw_only=True)
class ETSformerParameters:
    """
    ETSformer (Woo et al., 2022): exponential smoothing and frequency attention decompose each layer into level,
    growth and seasonality, which a decoder of damped growths and extrapolated seasons composes into the forecast.

    The decoder has one damping layer per encoder layer, so there is a single number of layers. The feed-forward
    networks use the sigmoid activation of the paper. Defaults follow the official implementation.

    Attributes
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of exponential smoothing and damping heads; must divide model_size.
    num_layers : int
        Number of encoder layers, and of decoder layers.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    top_k : int
        Number of Fourier components of largest amplitude kept by the frequency attention.
    dropout : float
        Dropout probability of the embedding, the layers and the summary head.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    model_size: int = 512
    num_heads: int = 8
    num_layers: int = 2
    feedforward_size: int = 2048
    top_k: int = 1
    dropout: float = 0.2

    def __post_init__(self) -> None:
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("feedforward_size", self.feedforward_size),
            ("top_k", self.top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
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
        return ETSformerForecastNetwork(self, shape)

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
        return ETSformerSummaryNetwork(self, shape)
