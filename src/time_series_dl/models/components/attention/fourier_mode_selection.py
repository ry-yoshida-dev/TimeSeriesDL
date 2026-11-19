from enum import Enum

import torch
from torch import Tensor


class FourierModeSelection(Enum):
    """
    Choice of the frequencies kept by the Fourier-enhanced blocks of FEDformer.

    The candidates are the ``max(1, length // 2)`` lowest frequencies of the real Fourier transform.

    Attributes
    ----------
    RANDOM
        Frequencies drawn at random once, when the block is built (the paper's default).
    LOWEST
        The lowest frequencies.
    """

    RANDOM = "random"
    LOWEST = "lowest"

    def select(self, length: int, modes: int) -> Tensor:
        """
        Select the kept frequencies of a sequence.

        Parameters
        ----------
        length : int
            Number of time steps of the sequence.
        modes : int
            Largest number of frequencies to keep.

        Returns
        -------
        Tensor
            The sorted indices of the kept frequencies, ``min(modes, max(1, length // 2))`` of them.

        Raises
        ------
        ValueError
            If length or modes is not positive.
        """
        if length <= 0 or modes <= 0:
            raise ValueError(f"length and modes must be positive, got {length} and {modes}")
        candidates: int = max(1, length // 2)
        count: int = min(modes, candidates)
        match self:
            case FourierModeSelection.RANDOM:
                return torch.randperm(candidates)[:count].sort().values
            case FourierModeSelection.LOWEST:
                return torch.arange(count)
