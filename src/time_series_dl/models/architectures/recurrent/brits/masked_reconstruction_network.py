from typing import override

from torch import Tensor

from ....interfaces import SequenceShape
from ....interfaces.masked_reconstruction import MaskedReconstruction
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from .recurrent_imputation import BRITSRecurrentImputation


class BRITSMaskedReconstructionNetwork(MaskedReconstructionNetwork):
    """
    BRITS (Cao et al., 2018): recurrent imputation run forward and backward in time, averaging the two directions'
    estimates.

    The network trains with its own loss alone: the estimation losses of both directions on the observed values plus
    the weighted mean absolute difference between the two directions' estimates (the consistency loss). The task's
    loss on the hidden values is not used for training.

    Parameters
    ----------
    hidden_size : int
        Size of the hidden state of each direction.
    consistency_weight : float
        Weight of the consistency loss.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, hidden_size: int, consistency_weight: float, shape: SequenceShape) -> None:
        super().__init__()
        self.consistency_weight: float = consistency_weight
        self.forward_imputation: BRITSRecurrentImputation = BRITSRecurrentImputation(shape.channels, hidden_size)
        self.backward_imputation: BRITSRecurrentImputation = BRITSRecurrentImputation(shape.channels, hidden_size)

    @property
    @override
    def hidden_value_loss_weight(self) -> float:
        return 0.0

    @override
    def forward(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        forward_estimates, forward_loss = self.forward_imputation(values, is_observed)
        reversed_estimates, backward_loss = self.backward_imputation(values.flip(1), is_observed.flip(1))
        backward_estimates: Tensor = reversed_estimates.flip(1)
        consistency_loss: Tensor = (forward_estimates - backward_estimates).abs().mean()
        return MaskedReconstruction(
            values=(forward_estimates + backward_estimates) / 2.0,
            loss=forward_loss + backward_loss + self.consistency_weight * consistency_loss,
        )
