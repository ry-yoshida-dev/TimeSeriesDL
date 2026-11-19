import math
from dataclasses import dataclass

from ....interfaces import SequenceShape
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from .masked_reconstruction_network import BRITSMaskedReconstructionNetwork


@dataclass(frozen=True, kw_only=True)
class BRITSParameters:
    """
    BRITS (Cao et al., 2018): bidirectional recurrent imputation with temporal decay, feature regression and a
    consistency loss between the two directions.

    The defaults follow the official implementation.

    Attributes
    ----------
    hidden_size : int
        Size of the LSTM hidden state of each direction.
    consistency_weight : float
        Weight of the consistency loss between the forward and backward estimates.

    Raises
    ------
    ValueError
        If hidden_size is not positive or consistency_weight is negative or infinite.
    """

    hidden_size: int = 64
    consistency_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if not (math.isfinite(self.consistency_weight) and self.consistency_weight >= 0.0):
            raise ValueError(f"consistency_weight must be finite and non-negative, got {self.consistency_weight}")

    def build_masked_reconstruction_network(self, shape: SequenceShape) -> MaskedReconstructionNetwork:
        """
        Build an untrained masked reconstruction network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences.

        Returns
        -------
        MaskedReconstructionNetwork
            The network.
        """
        return BRITSMaskedReconstructionNetwork(self.hidden_size, self.consistency_weight, shape)
