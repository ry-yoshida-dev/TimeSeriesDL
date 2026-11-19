from typing import override

from torch import Tensor

from ...models import ReconstructionNetwork
from ...models.interfaces.masked_reconstruction import MaskedReconstruction
from ...models.interfaces.masked_reconstruction_network import MaskedReconstructionNetwork


class ZeroFilledNetwork(MaskedReconstructionNetwork):
    """
    Lets a reconstruction network that does not read the observed mask impute: missing values are set to zero, and
    the network is trained only on the values the task hides.

    Parameters
    ----------
    network : ReconstructionNetwork
        The network that reconstructs zero-filled sequences.
    """

    def __init__(self, network: ReconstructionNetwork) -> None:
        super().__init__()
        self.network: ReconstructionNetwork = network

    @property
    @override
    def hidden_value_loss_weight(self) -> float:
        return 1.0

    @override
    def forward(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        return MaskedReconstruction(values=self.network(values * is_observed), loss=None)
