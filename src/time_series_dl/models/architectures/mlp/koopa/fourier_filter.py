from typing import cast

import torch
from torch import Tensor, nn


class KoopaFourierFilter(nn.Module):
    """
    Fourier filter of Koopa: the frequencies with the largest mean amplitude form the time-invariant part of a series
    and the others its time-variant part.

    The frequencies are ranked by the mean amplitude over the windows and channels of the first batch seen in training
    mode, and kept fixed afterwards (they are saved with the network); until then each batch ranks its own frequencies.

    Parameters
    ----------
    length : int
        Number of time steps of the series.
    invariant_fraction : float
        Fraction of the frequencies that are time-invariant.

    Raises
    ------
    ValueError
        If length is not positive or invariant_fraction is not in [0, 1].
    """

    def __init__(self, length: int, invariant_fraction: float) -> None:
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if not 0.0 <= invariant_fraction <= 1.0:
            raise ValueError(f"invariant_fraction must be in [0, 1], got {invariant_fraction}")
        super().__init__()
        self.length: int = length
        num_frequencies: int = length // 2 + 1
        self.num_invariant: int = int(num_frequencies * invariant_fraction)
        self.invariant_mask: Tensor
        self.register_buffer("invariant_mask", torch.zeros(num_frequencies))
        self.is_initialized: Tensor
        self.register_buffer("is_initialized", torch.tensor(False))

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """
        Split a batch of series.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        tuple[Tensor, Tensor]
            The time-variant and the time-invariant parts, each with the shape of ``values``.
        """
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(values, dim=1))
        mask: Tensor = self.invariant_mask
        if not bool(self.is_initialized):
            mask = self._rank(spectrum)
            if self.training:
                with torch.no_grad():
                    _ = self.invariant_mask.copy_(mask)
                    _ = self.is_initialized.fill_(True)
        variant: Tensor = cast(Tensor, torch.fft.irfft(spectrum * (1.0 - mask).unsqueeze(-1), n=self.length, dim=1))
        return variant, values - variant

    def _rank(self, spectrum: Tensor) -> Tensor:
        amplitude: Tensor = spectrum.detach().abs().mean(dim=(0, 2))
        mask: Tensor = torch.zeros_like(amplitude)
        _ = mask.scatter_(0, amplitude.topk(self.num_invariant).indices, 1.0)
        return mask

    def __call__(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values))
