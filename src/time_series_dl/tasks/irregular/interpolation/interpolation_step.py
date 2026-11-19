from dataclasses import dataclass

import torch
from torch import Tensor

from ....models.interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from ....models.interfaces.irregular_tensors import IrregularTensors
from .context_selection import ContextSelection


@dataclass(frozen=True, eq=False)
class IrregularInterpolationStep:
    """
    Training loss of an irregular interpolation network: each batch is split into context and targets, and the
    network computes its own loss of the targets.

    Attributes
    ----------
    network : IrregularInterpolationNetwork
        The network.
    context_selection : ContextSelection
        How each sequence is split.
    context_ratio : float
        Fraction of the observed values, or of the time range, that is context.
    """

    network: IrregularInterpolationNetwork
    context_selection: ContextSelection
    context_ratio: float

    def compute_loss(self, batch: tuple[Tensor, ...], is_training: bool) -> Tensor:
        """
        Compute the loss of a batch.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            The scaled times, values, observed mask and lengths, and a fixed random selection of context values.
        is_training : bool
            Whether a new random selection replaces the fixed one.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        sequences: IrregularTensors = IrregularTensors.from_batch(batch)
        is_drawn: Tensor = torch.rand_like(sequences.values) < self.context_ratio if is_training else batch[4]
        context, targets = self.context_selection.split(sequences, self.context_ratio, is_drawn)
        return self.network.compute_training_loss(context, targets)
