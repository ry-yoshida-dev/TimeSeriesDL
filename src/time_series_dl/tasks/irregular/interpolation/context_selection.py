from enum import Enum

import torch
from torch import Tensor

from ....models.interfaces.irregular_tensors import IrregularTensors


class ContextSelection(Enum):
    """
    How training splits the observations of each sequence into the context that a network reads and the targets it
    predicts.

    Attributes
    ----------
    RANDOM_OBSERVATIONS
        Interpolation: a random fraction of the observed values is the context, and every observed value is a
        target, as in the interpolation experiments of mTAN and the Latent ODE.
    EARLIER_TIMES
        Forecasting: the observations in the first fraction of each sequence's time range are the context, and the
        later ones are the targets, as in the extrapolation experiments of the Latent ODE.
    """

    RANDOM_OBSERVATIONS = "random_observations"
    EARLIER_TIMES = "earlier_times"

    def split(
        self, sequences: IrregularTensors, context_ratio: float, is_drawn: Tensor
    ) -> tuple[IrregularTensors, IrregularTensors]:
        """
        Split the observations into the context and the targets.

        Parameters
        ----------
        sequences : IrregularTensors
            The sequences.
        context_ratio : float
            Fraction of the observed values, or of the time range, that is context.
        is_drawn : Tensor
            Boolean draws with the shape of ``sequences.values``, True with probability ``context_ratio``; used by
            ``RANDOM_OBSERVATIONS``. The first observed value of each sequence is always context.

        Returns
        -------
        tuple[IrregularTensors, IrregularTensors]
            The context and the targets.
        """
        match self:
            case ContextSelection.RANDOM_OBSERVATIONS:
                flat_mask: Tensor = sequences.observed_mask.flatten(start_dim=1)
                is_first: Tensor = ((flat_mask.cumsum(dim=1) == 1) & (flat_mask > 0)).view_as(is_drawn)
                return sequences.select(is_drawn | is_first), sequences.trimmed()
            case ContextSelection.EARLIER_TIMES:
                first_time: Tensor = sequences.times[:, :1]
                last_index: Tensor = (sequences.lengths - 1).clamp(min=0).unsqueeze(1).to(torch.int64)
                last_time: Tensor = sequences.times.gather(1, last_index)
                split_time: Tensor = first_time + context_ratio * (last_time - first_time)
                is_context: Tensor = (sequences.times <= split_time).unsqueeze(-1)
                return sequences.select(is_context), sequences.select(~is_context)
