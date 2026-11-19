from dataclasses import dataclass

from ....components.layers import Activation
from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import TimeXerForecastNetwork


@dataclass(frozen=True, kw_only=True)
class TimeXerParameters:
    """
    TimeXer (Wang et al., NeurIPS 2024): patch tokens and a global token of each endogenous series, which attends to
    one token per exogenous series.

    Forecasting only. Defaults follow the official ETTh1 script with exogenous variables and horizon 96
    (thuml/TimeXer ``scripts/forecast_exogenous/ETTh1/TimeXer.sh`` with the defaults of ``run.py``).

    Attributes
    ----------
    patch_length : int
        Number of time steps of each non-overlapping patch of the endogenous series.
    model_size : int
        Size of each token.
    num_heads : int
        Number of attention heads; must divide model_size.
    num_layers : int
        Number of encoder layers.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    dropout : float
        Dropout probability of the embeddings, attention weights, sublayer outputs and head.
    activation : Activation
        Activation of the feed-forward networks.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, or dropout is not in [0, 1).
    """

    patch_length: int = 16
    model_size: int = 512
    num_heads: int = 8
    num_layers: int = 2
    feedforward_size: int = 512
    dropout: float = 0.1
    activation: Activation = Activation.GELU

    def __post_init__(self) -> None:
        for name, value in (
            ("patch_length", self.patch_length),
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("feedforward_size", self.feedforward_size),
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
        return TimeXerForecastNetwork(self, shape)
