from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....components.adversarial import FrozenModule
from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from .decoder import TranADDecoder
from .encoder import TranADEncoder

if TYPE_CHECKING:
    from .parameters import TranADParameters


class TranADAnomalyScoringNetwork(AnomalyScoringNetwork):
    """
    TranAD: an encoder E of the window W and a focus score F, and two decoders D1 and D2.

    Phase 1 reconstructs O1 = D1(E(W, 0)) and O2 = D2(E(W, 0)); phase 2 conditions on the focus score
    F = (O1 - W)^2 and reconstructs O2' = D2(E(W, F)). With w = 1/n in epoch n (counted from 1), the encoder and D1
    minimize L1 = w |O1 - W|^2 + (1 - w) |O2' - W|^2 (D1 learns to fool D2), and the encoder and D2 minimize
    L2 = w |O2 - W|^2 - (1 - w) |O2' - W|^2 (D2 learns to expose D1's reconstruction). ``compute_loss`` is one loss
    whose gradient equals that of L1 for D1, of L2 for D2 and of L1 + L2 for the encoder: the adversarial term is
    evaluated once through D1 with E and D2 frozen and once through D2 with E frozen and the focus score detached, so
    the term cancels for the encoder. Its value, which the trainer monitors, is the unweighted first-phase
    reconstruction error |O1 - W|^2 + |O2 - W|^2, since the paper's phase losses shift with the epoch.

    The decoders end with a sigmoid, as in the paper, so the windows should be scaled to [0, 1], for example by
    ``MinMaxScalingParameters`` as the detector's scaling; the sigmoid also bounds the adversarial term.

    The anomaly score of each step is (|O1 - W|^2 + |O2' - W|^2) / 2 averaged over channels.

    Parameters
    ----------
    parameters : TranADParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the windows.
    """

    def __init__(self, parameters: TranADParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.encoder: TranADEncoder = TranADEncoder(
            shape.channels, shape.length, parameters.feedforward_size, parameters.num_layers, parameters.dropout
        )
        self.first_decoder: TranADDecoder = TranADDecoder(
            shape.channels, parameters.feedforward_size, parameters.num_layers, parameters.dropout
        )
        self.second_decoder: TranADDecoder = TranADDecoder(
            shape.channels, parameters.feedforward_size, parameters.num_layers, parameters.dropout
        )

    def reconstruct(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the reconstructions of both phases.

        Parameters
        ----------
        values : Tensor
            Windows with shape (batch, length, channels).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            O1 and O2 of the first phase and O2' of the second, each with the shape of the windows.
        """
        memory: Tensor = self.encoder(values, torch.zeros_like(values))
        first: Tensor = self.first_decoder(values, memory)
        second: Tensor = self.second_decoder(values, memory)
        conditioned: Tensor = self.second_decoder(values, self.encoder(values, (first - values).square()))
        return first, second, conditioned

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        weight: float = 1.0 / (epoch + 1)
        memory: Tensor = self.encoder(values, torch.zeros_like(values))
        first: Tensor = self.first_decoder(values, memory)
        second: Tensor = self.second_decoder(values, memory)
        frozen_encoder: FrozenModule = FrozenModule(self.encoder)
        fooling_focus: Tensor = (self.first_decoder(values, memory.detach()) - values).square()
        fooling: Tensor = FrozenModule(self.second_decoder)(values, frozen_encoder(values, fooling_focus))
        exposing: Tensor = self.second_decoder(values, frozen_encoder(values, (first - values).square().detach()))
        reconstruction: Tensor = (first - values).square().mean() + (second - values).square().mean()
        adversarial: Tensor = (fooling - values).square().mean() - (exposing - values).square().mean()
        phases: Tensor = weight * reconstruction + (1.0 - weight) * adversarial
        return phases - phases.detach() + reconstruction.detach()

    @override
    def forward(self, values: Tensor) -> Tensor:
        first: Tensor
        conditioned: Tensor
        first, _, conditioned = self.reconstruct(values)
        return 0.5 * ((first - values).square() + (conditioned - values).square()).mean(dim=2)
