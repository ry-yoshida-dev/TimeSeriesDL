import math
from typing import cast

import torch
from torch import Tensor, nn


class ETSformerFrequencyAttention(nn.Module):
    """
    Frequency attention (FA) of ETSformer: the Fourier components of largest amplitude of each feature, extrapolated
    beyond the input as sinusoids.

    The mean and, for even lengths, the Nyquist component are left out. Sequences too short to have any other
    component have no seasonality.

    Parameters
    ----------
    top_k : int
        Number of components kept per feature; fewer if the sequence has fewer.
    horizon : int
        Number of steps extrapolated after the input.

    Raises
    ------
    ValueError
        If top_k is not positive or horizon is negative.
    """

    def __init__(self, top_k: int, horizon: int) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if horizon < 0:
            raise ValueError(f"horizon must not be negative, got {horizon}")
        super().__init__()
        self.top_k: int = top_k
        self.horizon: int = horizon

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Extract the seasonality of a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Seasonal representations with shape (batch, length + horizon, model_size).
        """
        batch, length, model_size = tokens.shape
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(tokens, dim=1))
        end: int = spectrum.shape[1] - 1 if length % 2 == 0 else spectrum.shape[1]
        spectrum = spectrum[:, 1:end]
        num_selected: int = min(self.top_k, spectrum.shape[1])
        if num_selected == 0:
            return tokens.new_zeros(batch, length + self.horizon, model_size)
        frequencies: Tensor = cast(Tensor, torch.fft.rfftfreq(length, device=tokens.device, dtype=tokens.dtype))[1:end]
        amplitudes: Tensor
        indices: Tensor
        amplitudes, indices = torch.topk(spectrum.abs(), num_selected, dim=1)
        phases: Tensor = torch.gather(spectrum.angle(), 1, indices)
        times: Tensor = torch.arange(length + self.horizon, device=tokens.device, dtype=tokens.dtype).view(1, 1, -1, 1)
        # Each component and its complex conjugate together contribute twice the real part.
        waves: Tensor = (
            (2.0 / length)
            * amplitudes.unsqueeze(2)
            * torch.cos(2.0 * math.pi * frequencies[indices].unsqueeze(2) * times + phases.unsqueeze(2))
        )
        return waves.sum(dim=1)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
