from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

import torch
from deep_learning_loss import MAELoss
from torch import Tensor, nn

from ....interfaces import SequenceShape
from ....interfaces.masked_reconstruction import MaskedReconstruction
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from .block import SAITSBlock

if TYPE_CHECKING:
    from .parameters import SAITSParameters

_ABSOLUTE_ERROR: Final[MAELoss] = MAELoss()


class SAITSMaskedReconstructionNetwork(MaskedReconstructionNetwork):
    """
    SAITS (Du et al., 2023): two diagonally masked self-attention blocks, the second refining the first block's
    imputation, combined by weights learned from the observed mask and the attention map.

    The network's own loss is the observed reconstruction task (ORT): the mean absolute error of the three estimates
    on the observed inputs. The task's loss on the hidden values is the masked imputation task (MIT) on the combined
    estimate.

    Parameters
    ----------
    parameters : SAITSParameters
        The architecture and the weights of the two tasks.
    shape : SequenceShape
        The dimensions of the sequences.

    Raises
    ------
    ValueError
        If the sequences have a single time step.
    """

    def __init__(self, parameters: SAITSParameters, shape: SequenceShape) -> None:
        if shape.length < 2:
            raise ValueError(f"SAITS needs at least two time steps, got {shape.length}")
        super().__init__()
        self.reconstruction_weight: float = parameters.observed_reconstruction_weight
        self.imputation_weight: float = parameters.masked_imputation_weight
        self.first_block: SAITSBlock = SAITSBlock(parameters, shape.length, shape.channels)
        self.first_projection: nn.Linear = nn.Linear(parameters.model_size, shape.channels)
        self.second_block: SAITSBlock = SAITSBlock(parameters, shape.length, shape.channels)
        self.second_projection: nn.Sequential = nn.Sequential(
            nn.Linear(parameters.model_size, shape.channels), nn.ReLU(), nn.Linear(shape.channels, shape.channels)
        )
        self.combination: nn.Linear = nn.Linear(shape.length + shape.channels, shape.channels)

    @property
    @override
    def hidden_value_loss_weight(self) -> float:
        return self.imputation_weight

    @override
    def forward(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        mask: Tensor = is_observed.to(values.dtype)
        first_tokens, _ = self.first_block(values, mask)
        first: Tensor = self.first_projection(first_tokens)
        replaced: Tensor = mask * values + (1.0 - mask) * first
        second_tokens, weights = self.second_block(replaced, mask)
        second: Tensor = self.second_projection(second_tokens)
        combining_weights: Tensor = torch.sigmoid(self.combination(torch.cat([mask, weights], dim=2)))
        combined: Tensor = (1.0 - combining_weights) * second + combining_weights * first
        reconstruction_loss: Tensor = (
            _ABSOLUTE_ERROR.compute_masked(first, values, is_observed)
            + _ABSOLUTE_ERROR.compute_masked(second, values, is_observed)
            + _ABSOLUTE_ERROR.compute_masked(combined, values, is_observed)
        ) / 3.0
        return MaskedReconstruction(values=combined, loss=reconstruction_loss * self.reconstruction_weight)
