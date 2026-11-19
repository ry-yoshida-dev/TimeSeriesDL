from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from torch import Tensor, nn

from ....components.adversarial import FrozenModule
from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork

if TYPE_CHECKING:
    from .parameters import USADParameters


class USADAnomalyScoringNetwork(AnomalyScoringNetwork):
    """
    Shared encoder E and decoders D1 and D2 over the flattened window W, with AE1 = D1(E(W)) and AE2 = D2(E(W)).

    In epoch n (counted from 1), AE1 minimizes (1/n)|W - AE1(W)|^2 + (1 - 1/n)|W - AE2(AE1(W))|^2 and AE2 minimizes
    (1/n)|W - AE2(W)|^2 - (1 - 1/n)|W - AE2(AE1(W))|^2: AE1 learns to fool AE2, AE2 to tell real windows from AE1's
    reconstructions. The paper updates the two autoencoders with two optimizers; ``compute_loss`` is one loss whose
    gradient equals the sum of both phase gradients at the current weights: the encoder gets both, D1 only AE1's and
    D2 only AE2's, because the adversarial term is evaluated once with E and D2 frozen and once with E and the input
    frozen. Its value is the unweighted sum of the two reconstruction errors |W - AE1(W)|^2 + |W - AE2(W)|^2, which
    the trainer monitors, since the paper's phase losses shift with the epoch and their adversarial terms cancel.

    The decoders end with a sigmoid, as in the paper, so the windows should be scaled to [0, 1], for example by
    ``MinMaxScalingParameters`` as the detector's scaling; the sigmoid also bounds the adversarial term.

    Each step's anomaly score is alpha |W - AE1(W)|^2 + beta |W - AE2(AE1(W))|^2, averaged over channels.

    Parameters
    ----------
    parameters : USADParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the windows.
    """

    def __init__(self, parameters: USADParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.alpha: float = parameters.alpha
        self.beta: float = parameters.beta
        window_size: int = shape.length * shape.channels
        latent_size: int = shape.length * parameters.latent_size_per_step
        half: int = max(window_size // 2, 1)
        quarter: int = max(window_size // 4, 1)
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size, half),
            nn.ReLU(),
            nn.Linear(half, quarter),
            nn.ReLU(),
            nn.Linear(quarter, latent_size),
            nn.ReLU(),
        )
        self.first_decoder: nn.Sequential = self._build_decoder(latent_size, quarter, half, shape)
        self.second_decoder: nn.Sequential = self._build_decoder(latent_size, quarter, half, shape)

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        weight: float = 1.0 / (epoch + 1)
        latent: Tensor = self.encoder(values)
        first: Tensor = self.first_decoder(latent)
        second: Tensor = self.second_decoder(latent)
        frozen_encoder: FrozenModule = FrozenModule(self.encoder)
        # The adversarial term |W - AE2(AE1(W))|^2: minimized through D1, maximized through D2.
        fooling: Tensor = FrozenModule(self.second_decoder)(frozen_encoder(self.first_decoder(latent.detach())))
        discriminating: Tensor = self.second_decoder(frozen_encoder(first.detach()))
        reconstruction: Tensor = (values - first).square().mean() + (values - second).square().mean()
        adversarial: Tensor = (values - fooling).square().mean() - (values - discriminating).square().mean()
        phases: Tensor = weight * reconstruction + (1.0 - weight) * adversarial
        return phases - phases.detach() + reconstruction.detach()

    @override
    def forward(self, values: Tensor) -> Tensor:
        first: Tensor = self.first_decoder(self.encoder(values))
        both: Tensor = self.second_decoder(self.encoder(first))
        scores: Tensor = self.alpha * (values - first).square() + self.beta * (values - both).square()
        return scores.mean(dim=2)

    @staticmethod
    def _build_decoder(latent_size: int, quarter: int, half: int, shape: SequenceShape) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(latent_size, quarter),
            nn.ReLU(),
            nn.Linear(quarter, half),
            nn.ReLU(),
            nn.Linear(half, shape.length * shape.channels),
            nn.Sigmoid(),
            nn.Unflatten(1, (shape.length, shape.channels)),
        )

    def reconstruct(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the three reconstructions of a batch of windows.

        Parameters
        ----------
        values : Tensor
            Windows with shape (batch, length, channels).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            AE1(W), AE2(W) and AE2(AE1(W)), each with the shape of the windows.
        """
        latent: Tensor = self.encoder(values)
        first: Tensor = cast(Tensor, self.first_decoder(latent))
        return first, cast(Tensor, self.second_decoder(latent)), cast(Tensor, self.second_decoder(self.encoder(first)))
