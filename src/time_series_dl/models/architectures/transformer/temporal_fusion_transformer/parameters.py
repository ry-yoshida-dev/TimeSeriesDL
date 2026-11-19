from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import TemporalFusionTransformerForecastNetwork


@dataclass(frozen=True, kw_only=True)
class TemporalFusionTransformerParameters:
    """
    Temporal Fusion Transformer (Lim et al., 2021): variable selection, an LSTM sequence-to-sequence layer and
    interpretable multi-head attention, joined by gated skip connections.

    This package has no static covariates and no known future inputs, so static covariate encoders and static
    enrichment are omitted, and the decoder reads learned embeddings of the horizon positions. The defaults are those
    of the reference implementation in neuralforecast (hidden size 128, 4 heads, dropout 0.1, one LSTM layer).

    Attributes
    ----------
    hidden_size : int
        Size of the vectors of every layer.
    num_heads : int
        Number of attention heads; must divide hidden_size.
    num_lstm_layers : int
        Number of stacked layers of the LSTM encoder and of the LSTM decoder.
    dropout : float
        Dropout probability of the gated layers, the LSTM layers and the attention weights.

    Raises
    ------
    ValueError
        If a size is not positive, hidden_size is not divisible by num_heads or dropout is not in [0, 1).
    """

    hidden_size: int = 128
    num_heads: int = 4
    num_lstm_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_heads", self.num_heads),
            ("num_lstm_layers", self.num_lstm_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")
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
        return TemporalFusionTransformerForecastNetwork(self, shape)
