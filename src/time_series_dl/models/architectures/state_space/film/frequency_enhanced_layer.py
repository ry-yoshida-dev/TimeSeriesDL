from typing import cast

import torch
from torch import Tensor, nn


class FiLMFrequencyEnhancedLayer(nn.Module):
    """
    Frequency enhanced layer of FiLM: a learned complex linear map between memory dimensions at each of the lowest
    Fourier modes along time, with the higher modes dropped.

    Parameters
    ----------
    state_size : int
        Size of each memory state.
    length : int
        Number of memory states along time.
    num_modes : int
        Largest number of lowest Fourier modes kept.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, state_size: int, length: int, num_modes: int) -> None:
        for name, value in (("state_size", state_size), ("length", length), ("num_modes", num_modes)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        super().__init__()
        self.length: int = length
        self.num_modes: int = min(num_modes, length // 2 + 1)
        self.weights: nn.Parameter = nn.Parameter(
            torch.rand(state_size, state_size, self.num_modes, 2) / (state_size * state_size)
        )

    def forward(self, states: Tensor) -> Tensor:
        """
        Filter the memory states along time.

        Parameters
        ----------
        states : Tensor
            Memory states with shape (batch, channels, state_size, length).

        Returns
        -------
        Tensor
            Filtered states with the shape of ``states``.
        """
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(states, dim=-1))
        mixed: Tensor = torch.einsum(
            "bcix,iox->bcox", spectrum[..., : self.num_modes], torch.view_as_complex(self.weights)
        )
        dropped: Tensor = mixed.new_zeros(*mixed.shape[:-1], spectrum.shape[-1] - self.num_modes)
        filtered: Tensor = cast(Tensor, torch.fft.irfft(torch.cat([mixed, dropped], dim=-1), n=self.length, dim=-1))
        return filtered

    def __call__(self, states: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(states))
