from dataclasses import dataclass

from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from .anomaly_scoring_network import DCdetectorAnomalyScoringNetwork


@dataclass(frozen=True, kw_only=True)
class DCdetectorParameters:
    """
    DCdetector (Yang et al., 2023): dual attention over a patch-wise and an in-patch view of each channel, trained
    with a purely contrastive loss on the discrepancy of the two views' attention, and scored by that discrepancy.

    The defaults follow the official scripts for MSL.

    Attributes
    ----------
    patch_sizes : tuple[int, ...]
        Patch length of each scale; each must divide the window length.
    model_size : int
        Size of each token.
    num_heads : int
        Number of attention heads; must divide model_size.
    num_layers : int
        Number of dual attention layers per scale, each attending over the same embedded tokens.
    dropout : float
        Dropout probability of the embeddings and the attention.
    score_temperature : float
        Factor of the discrepancy inside the softmax of the anomaly score.

    Raises
    ------
    ValueError
        If patch_sizes is empty or repeats a size, a size or score_temperature is not positive, num_heads does not
        divide model_size, or dropout is not in [0, 1).
    """

    patch_sizes: tuple[int, ...] = (3, 5)
    model_size: int = 256
    num_heads: int = 1
    num_layers: int = 3
    dropout: float = 0.0
    score_temperature: float = 50.0

    def __post_init__(self) -> None:
        if not self.patch_sizes:
            raise ValueError("patch_sizes must not be empty.")
        if len(set(self.patch_sizes)) != len(self.patch_sizes):
            raise ValueError(f"patch_sizes must not repeat, got {self.patch_sizes}")
        if any(size <= 0 for size in self.patch_sizes):
            raise ValueError(f"patch_sizes must be positive, got {self.patch_sizes}")
        for name, value in (
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_size % self.num_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must divide model_size {self.model_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.score_temperature <= 0.0:
            raise ValueError(f"score_temperature must be positive, got {self.score_temperature}")

    def build_anomaly_scoring_network(self, shape: SequenceShape) -> AnomalyScoringNetwork:
        """
        Build an untrained anomaly scoring network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the windows; every patch size must divide the length.

        Returns
        -------
        AnomalyScoringNetwork
            The network.

        Raises
        ------
        ValueError
            If a patch size does not divide the window length.
        """
        return DCdetectorAnomalyScoringNetwork(self, shape)
