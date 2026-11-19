from typing import cast, override

import torch
from torch import Tensor, nn

from ..attention_mechanism import AttentionMechanism
from ..fourier_mode_selection import FourierModeSelection


class FourierAttentionMechanism(AttentionMechanism):
    """
    Fourier-enhanced block of FEDformer (Zhou et al., ICML 2022), which replaces self-attention.

    The queries are transformed to the frequency domain, the selected frequencies are mixed across the channels of
    each head by learned complex weights, the others are zeroed, and the result is transformed back. Keys and values
    are ignored.

    Parameters
    ----------
    num_heads : int
        Number of heads.
    head_size : int
        Number of channels per head.
    length : int
        Number of time steps of the sequences.
    modes : int
        Largest number of kept frequencies.
    selection : FourierModeSelection
        Choice of the kept frequencies.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(
        self, num_heads: int, head_size: int, length: int, modes: int, selection: FourierModeSelection
    ) -> None:
        if num_heads <= 0 or head_size <= 0:
            raise ValueError(f"num_heads and head_size must be positive, got {num_heads} and {head_size}")
        frequencies: Tensor = selection.select(length, modes)
        super().__init__()
        self.length: int = length
        self.frequencies: Tensor
        self.register_buffer("frequencies", frequencies)
        model_size: int = num_heads * head_size
        # Real and imaginary parts in the last dimension; uniform initialization scaled as in the reference code.
        self.weight: nn.Parameter = nn.Parameter(
            torch.rand(num_heads, head_size, head_size, frequencies.shape[0], 2) / (model_size * model_size)
        )

    @property
    @override
    def is_query_only(self) -> bool:
        return True

    @override
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        if queries.shape[2] != self.length:
            raise ValueError(f"expected sequences of length {self.length}, got {queries.shape[2]}")
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(queries, dim=2))
        selected: Tensor = spectrum.index_select(2, self.frequencies)
        mixed: Tensor = torch.einsum("bhmi,hiom->bhmo", selected, torch.view_as_complex(self.weight))
        output_spectrum: Tensor = torch.zeros_like(spectrum).index_copy(2, self.frequencies, mixed)
        outputs: Tensor = cast(Tensor, torch.fft.irfft(output_spectrum, n=self.length, dim=2))
        return outputs
