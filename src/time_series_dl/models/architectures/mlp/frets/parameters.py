from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import FreTSForecastNetwork


@dataclass(frozen=True, kw_only=True)
class FreTSParameters:
    """
    FreTS (Yi et al., 2023): frequency-domain MLPs that mix embedded values across channels and along time, followed by
    an MLP from each channel's window to its horizon.

    The defaults follow the official code.

    Attributes
    ----------
    embed_size : int
        Size of the embedding vector of each value.
    hidden_size : int
        Size of the hidden layer of the output MLP.
    sparsity_threshold : float
        Threshold of the soft shrinkage in the frequency MLPs.
    weight_scale : float
        Standard deviation of the initial weights of the frequency MLPs.
    is_channel_mixing : bool
        Whether the frequency channel learner mixes the channels before the temporal learner.

    Raises
    ------
    ValueError
        If a size or weight_scale is not positive, or sparsity_threshold is negative.
    """

    embed_size: int = 128
    hidden_size: int = 256
    sparsity_threshold: float = 0.01
    weight_scale: float = 0.02
    is_channel_mixing: bool = True

    def __post_init__(self) -> None:
        if self.embed_size <= 0:
            raise ValueError(f"embed_size must be positive, got {self.embed_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.sparsity_threshold < 0.0:
            raise ValueError(f"sparsity_threshold must not be negative, got {self.sparsity_threshold}")
        if self.weight_scale <= 0.0:
            raise ValueError(f"weight_scale must be positive, got {self.weight_scale}")

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
        return FreTSForecastNetwork(self, shape)
