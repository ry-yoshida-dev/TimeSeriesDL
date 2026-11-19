from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import SequenceShape
from ....interfaces.masked_reconstruction import MaskedReconstruction
from ....interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from .conditional_imputer import CSDIConditionalImputer

if TYPE_CHECKING:
    from .parameters import CSDIParameters


class CSDIMaskedReconstructionNetwork(MaskedReconstructionNetwork):
    """
    CSDI imputation: trained only with its own denoising loss, and imputing by the median of samples.

    While training, the conditional mask is drawn from the observed values by the paper's random strategy: each
    sequence draws a missing ratio uniformly from [0, 1) and that fraction of its observed values, chosen at random,
    become the targets of the denoising loss, the rest being conditioned on. The returned values are then the inputs
    unchanged, since ``hidden_value_loss_weight`` is zero. Outside training every observed value is conditioned on,
    and each missing value is the median of ``num_imputation_samples`` samples of the reverse diffusion.

    Parameters
    ----------
    parameters : CSDIParameters
        The model configuration and the number of imputation samples.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: CSDIParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.num_samples: int = parameters.num_imputation_samples
        self.imputer: CSDIConditionalImputer = parameters.build_conditional_imputer(shape.channels)

    @property
    @override
    def hidden_value_loss_weight(self) -> float:
        return 0.0

    @override
    def forward(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        # The imputer reads (batch, features, length) arrays; the task passes (batch, length, channels).
        arrays: Tensor = values.transpose(1, 2)
        observed_mask: Tensor = is_observed.transpose(1, 2).to(values.dtype)
        if self.training:
            conditional_mask: Tensor = self._draw_conditional_mask(observed_mask)
            loss: Tensor = self.imputer.compute_loss(arrays, observed_mask, conditional_mask)
            return MaskedReconstruction(values=values, loss=loss)
        samples: Tensor = self.imputer.impute(arrays, observed_mask, self.num_samples)
        return MaskedReconstruction(values=samples.median(dim=1).values.transpose(1, 2), loss=None)

    @staticmethod
    def _draw_conditional_mask(observed_mask: Tensor) -> Tensor:
        batch: int = observed_mask.shape[0]
        flat: Tensor = observed_mask.reshape(batch, -1)
        # Observed entries get uniform scores and missing ones -1, so ranking the scores in descending order puts
        # the observed entries first in random order.
        scores: Tensor = torch.where(flat > 0, torch.rand_like(flat), torch.full_like(flat, -1.0))
        order: Tensor = scores.argsort(dim=1, descending=True)
        ranks: Tensor = torch.empty_like(order).scatter_(
            1, order, torch.arange(flat.shape[1], device=flat.device).expand_as(order)
        )
        ratios: Tensor = torch.rand(batch, 1, device=flat.device, dtype=flat.dtype)
        num_targets: Tensor = torch.round(flat.sum(dim=1, keepdim=True) * ratios)
        is_conditioned: Tensor = (flat > 0) & (ranks >= num_targets)
        return is_conditioned.to(flat.dtype).reshape(observed_mask.shape)
