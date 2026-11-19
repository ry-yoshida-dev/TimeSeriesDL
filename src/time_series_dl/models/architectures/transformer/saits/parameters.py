import math
from dataclasses import dataclass

from ....interfaces import SequenceShape
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from .masked_reconstruction_network import SAITSMaskedReconstructionNetwork


@dataclass(frozen=True, kw_only=True)
class SAITSParameters:
    """
    SAITS (Du et al., 2023): self-attention-based imputation with two diagonally masked self-attention blocks,
    trained jointly on the observed reconstruction task and the masked imputation task.

    The defaults follow the usage example of PyPOTS, the reference implementation by the paper's first author. For the
    paper's masked imputation loss, train with ``MAELoss``.

    Attributes
    ----------
    num_layers : int
        Number of encoder layers in each block.
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads.
    key_size : int
        Size of the queries and keys of each head.
    value_size : int
        Size of the values of each head.
    feedforward_size : int
        Hidden size of the feed-forward network of each layer.
    dropout : float
        Dropout probability of the embeddings and of each layer.
    attention_dropout : float
        Dropout probability of the attention weights.
    observed_reconstruction_weight : float
        Weight of the observed reconstruction task (ORT), the error on the observed inputs.
    masked_imputation_weight : float
        Weight of the masked imputation task (MIT), the task's loss on the observed values hidden from the network.

    Raises
    ------
    ValueError
        If a size is not positive, a dropout is not in [0, 1), or a loss weight is negative, infinite, or both are
        zero.
    """

    num_layers: int = 2
    model_size: int = 256
    num_heads: int = 4
    key_size: int = 64
    value_size: int = 64
    feedforward_size: int = 128
    dropout: float = 0.1
    attention_dropout: float = 0.0
    observed_reconstruction_weight: float = 1.0
    masked_imputation_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("num_layers", self.num_layers),
            ("model_size", self.model_size),
            ("num_heads", self.num_heads),
            ("key_size", self.key_size),
            ("value_size", self.value_size),
            ("feedforward_size", self.feedforward_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, probability in (("dropout", self.dropout), ("attention_dropout", self.attention_dropout)):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {probability}")
        for name, weight in (
            ("observed_reconstruction_weight", self.observed_reconstruction_weight),
            ("masked_imputation_weight", self.masked_imputation_weight),
        ):
            if not (math.isfinite(weight) and weight >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative, got {weight}")
        if self.observed_reconstruction_weight == 0.0 and self.masked_imputation_weight == 0.0:
            raise ValueError("observed_reconstruction_weight and masked_imputation_weight must not both be zero.")

    def build_masked_reconstruction_network(self, shape: SequenceShape) -> MaskedReconstructionNetwork:
        """
        Build an untrained masked reconstruction network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences; the length must exceed one.

        Returns
        -------
        MaskedReconstructionNetwork
            The network.

        Raises
        ------
        ValueError
            If the sequences have a single time step.
        """
        return SAITSMaskedReconstructionNetwork(self, shape)
