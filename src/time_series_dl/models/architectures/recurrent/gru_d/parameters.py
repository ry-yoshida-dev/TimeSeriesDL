from dataclasses import dataclass

from ....interfaces import SequenceShape
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from .masked_reconstruction_network import GRUDMaskedReconstructionNetwork


@dataclass(frozen=True, kw_only=True)
class GRUDParameters:
    """
    GRU-D (Che et al., 2018): a GRU with trainable decays of the inputs and of the hidden state, driven by the
    observed mask and the time since the last observations.

    Attributes
    ----------
    hidden_size : int
        Size of the GRU hidden state.

    Raises
    ------
    ValueError
        If hidden_size is not positive.
    """

    hidden_size: int = 64

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")

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
        return GRUDMaskedReconstructionNetwork(self.hidden_size, shape)
