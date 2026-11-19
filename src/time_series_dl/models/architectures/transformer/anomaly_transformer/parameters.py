from dataclasses import dataclass

from ....components.layers import Activation
from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from .anomaly_scoring_network import AnomalyTransformerAnomalyScoringNetwork


@dataclass(frozen=True, kw_only=True)
class AnomalyTransformerParameters:
    """
    Anomaly Transformer (Xu et al., 2022): anomaly-attention layers that learn a Gaussian prior association and a
    series association of every step, trained with a minimax association discrepancy and scored by the
    association-based criterion.

    The defaults follow the official scripts.

    Attributes
    ----------
    model_size : int
        Size of each step's token.
    num_heads : int
        Number of attention heads; must divide model_size.
    num_layers : int
        Number of anomaly-attention encoder layers.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    dropout : float
        Dropout probability of the embedding, the attention and the feed-forward networks.
    activation : Activation
        Activation of the feed-forward networks.
    discrepancy_weight : float
        Weight lambda of the association discrepancy in the minimax losses.
    score_temperature : float
        Factor of the association discrepancy inside the softmax of the anomaly score.

    Raises
    ------
    ValueError
        If a size is not positive, num_heads does not divide model_size, dropout is not in [0, 1), or a weight is
        not positive.
    """

    model_size: int = 512
    num_heads: int = 8
    num_layers: int = 3
    feedforward_size: int = 512
    dropout: float = 0.0
    activation: Activation = Activation.GELU
    discrepancy_weight: float = 3.0
    score_temperature: float = 50.0

    def __post_init__(self) -> None:
        for name, value in (
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
        if self.discrepancy_weight <= 0.0 or self.score_temperature <= 0.0:
            raise ValueError(
                "discrepancy_weight and score_temperature must be positive, got "
                + f"{self.discrepancy_weight} and {self.score_temperature}"
            )

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
        return AnomalyTransformerAnomalyScoringNetwork(self, shape)
