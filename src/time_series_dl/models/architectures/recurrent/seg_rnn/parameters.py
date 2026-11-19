from dataclasses import dataclass

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import SegRNNForecastNetwork


@dataclass(frozen=True, kw_only=True)
class SegRNNParameters:
    """
    SegRNN (Lin et al., 2023): a GRU over segments of each target series, decoded in parallel for every output segment
    from position and channel embeddings.

    The defaults are those of the paper (segment length 48, hidden size 512, dropout 0.5, with channel embeddings).

    Attributes
    ----------
    segment_length : int
        Number of time steps of each segment; at most the input length of the networks built.
    hidden_size : int
        Size of the segment embeddings and of the GRU state.
    dropout : float
        Dropout probability before the output layer.
    is_channel_embedded : bool
        Whether half of each decoder input embeds the target channel, the other half embedding the output segment's
        position; otherwise the whole input embeds the position.

    Raises
    ------
    ValueError
        If a size is not positive, dropout is not in [0, 1), or hidden_size is odd while channels are embedded.
    """

    segment_length: int = 48
    hidden_size: int = 512
    dropout: float = 0.5
    is_channel_embedded: bool = True

    def __post_init__(self) -> None:
        if self.segment_length <= 0:
            raise ValueError(f"segment_length must be positive, got {self.segment_length}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.is_channel_embedded and self.hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even when channels are embedded, got {self.hidden_size}")
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

        Raises
        ------
        ValueError
            If the segment is longer than the input window.
        """
        return SegRNNForecastNetwork(self, shape)
