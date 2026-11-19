from dataclasses import dataclass

from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from .anomaly_scoring_network import TranADAnomalyScoringNetwork


@dataclass(frozen=True, kw_only=True)
class TranADParameters:
    """
    TranAD (Tuli et al., 2022): a Transformer encoder over the window and a focus score with two decoders, trained in
    two phases with adversarial self-conditioning: the second phase feeds the first phase's reconstruction error back
    as the focus score.

    The token size is twice the number of channels (the window concatenated with the focus score) and the number of
    heads equals the number of channels, as in the official code. The other defaults follow the official code. The
    reconstructions end with a sigmoid, so the detector should scale the channels to [0, 1]
    (``MinMaxScalingParameters``), as in the paper.

    Attributes
    ----------
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    num_layers : int
        Number of encoder layers and of layers of each decoder.
    dropout : float
        Dropout probability of the position encoding, the attention and the feed-forward networks.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    feedforward_size: int = 16
    num_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.feedforward_size <= 0:
            raise ValueError(f"feedforward_size must be positive, got {self.feedforward_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_anomaly_scoring_network(self, shape: SequenceShape) -> AnomalyScoringNetwork:
        """
        Build an untrained anomaly scoring network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the windows.

        Returns
        -------
        AnomalyScoringNetwork
            The network.
        """
        return TranADAnomalyScoringNetwork(self, shape)
