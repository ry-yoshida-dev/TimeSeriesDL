from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class PeriodDetector:
    """
    Detects the dominant periods of a batch of sequences from the amplitudes of their Fourier spectrum.

    The frequencies are ranked by their amplitude averaged over the batch and the features; the constant component is
    never chosen. Frequency ``f`` of a sequence of length ``T`` gives the period ``T // f``, which is always in
    ``[2, T]``. A sequence too short to have any non-constant frequency gets the single period ``T``.

    Attributes
    ----------
    top_k : int
        Largest number of periods to detect.

    Raises
    ------
    ValueError
        If top_k is not positive.
    """

    top_k: int

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

    def detect(self, values: Tensor) -> tuple[list[int], Tensor]:
        """
        Detect the dominant periods.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, features).

        Returns
        -------
        tuple[list[int], Tensor]
            The periods, at most ``top_k``, and the amplitude of each period in each sequence with shape
            (batch, number of periods).
        """
        length: int = values.shape[1]
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(values, dim=1))
        amplitudes: Tensor = spectrum.abs()[:, 1:].mean(dim=2)
        count: int = min(self.top_k, amplitudes.shape[1])
        if count == 0:
            return [length], values.new_ones(values.shape[0], 1)
        indices: Tensor = amplitudes.detach().mean(dim=0).topk(count).indices
        ranks: list[int] = cast(list[int], indices.tolist())
        frequencies: list[int] = [rank + 1 for rank in ranks]
        return [length // frequency for frequency in frequencies], amplitudes[:, indices]
