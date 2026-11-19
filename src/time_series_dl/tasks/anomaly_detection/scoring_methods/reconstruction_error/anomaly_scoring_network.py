from typing import override

from torch import Tensor

from .....losses import PointLoss
from .....models import ReconstructionNetwork
from .....models.interfaces.anomaly_scoring_network import AnomalyScoringNetwork


class ReconstructionErrorScoringNetwork(AnomalyScoringNetwork):
    """
    Scores each step by the point loss of its reconstruction, averaged over channels, and trains on the mean loss.

    Parameters
    ----------
    network : ReconstructionNetwork
        The reconstruction network of the model.
    loss : PointLoss
        The loss of each reconstructed value.
    """

    def __init__(self, network: ReconstructionNetwork, loss: PointLoss) -> None:
        super().__init__()
        self.network: ReconstructionNetwork = network
        self.loss: PointLoss = loss

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        return self.loss.compute(self.network(values), values)

    @override
    def forward(self, values: Tensor) -> Tensor:
        return self.loss.compute_elementwise(self.network(values), values).mean(dim=2)
