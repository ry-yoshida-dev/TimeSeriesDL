from typing import cast, override

import torch
from torch import Tensor, nn

from ..attention_mechanism import AttentionMechanism
from ..fourier_mode_selection import FourierModeSelection


class FourierCrossAttentionMechanism(AttentionMechanism):
    """
    Fourier-enhanced cross-attention of FEDformer (Zhou et al., ICML 2022).

    Queries, keys and values are transformed to the frequency domain and reduced to their selected frequencies. The
    complex attention ``tanh(Q K^T) V`` is computed there, mixed across the channels of each head by learned complex
    weights, placed back at the query frequencies and transformed back.

    Parameters
    ----------
    num_heads : int
        Number of heads.
    head_size : int
        Number of channels per head.
    query_length : int
        Number of time steps of the queries.
    key_length : int
        Number of time steps of the keys and values.
    modes : int
        Largest number of kept frequencies of each sequence.
    selection : FourierModeSelection
        Choice of the kept frequencies.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        query_length: int,
        key_length: int,
        modes: int,
        selection: FourierModeSelection,
    ) -> None:
        if num_heads <= 0 or head_size <= 0:
            raise ValueError(f"num_heads and head_size must be positive, got {num_heads} and {head_size}")
        query_frequencies: Tensor = selection.select(query_length, modes)
        key_frequencies: Tensor = selection.select(key_length, modes)
        super().__init__()
        self.query_length: int = query_length
        self.key_length: int = key_length
        self.query_frequencies: Tensor
        self.register_buffer("query_frequencies", query_frequencies)
        self.key_frequencies: Tensor
        self.register_buffer("key_frequencies", key_frequencies)
        model_size: int = num_heads * head_size
        # Real and imaginary parts in the last dimension; uniform initialization scaled as in the reference code.
        self.weight: nn.Parameter = nn.Parameter(
            torch.rand(num_heads, head_size, head_size, query_frequencies.shape[0], 2) / (model_size * model_size)
        )

    @override
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        if queries.shape[2] != self.query_length or keys.shape[2] != self.key_length:
            raise ValueError(
                f"expected {self.query_length} queries and {self.key_length} keys, "
                + f"got {queries.shape[2]} and {keys.shape[2]}"
            )
        query_spectrum: Tensor = cast(Tensor, torch.fft.rfft(queries, dim=2))
        selected_queries: Tensor = query_spectrum.index_select(2, self.query_frequencies)
        selected_keys: Tensor = cast(Tensor, torch.fft.rfft(keys, dim=2)).index_select(2, self.key_frequencies)
        selected_values: Tensor = cast(Tensor, torch.fft.rfft(values, dim=2)).index_select(2, self.key_frequencies)
        scores: Tensor = torch.einsum("bhxe,bhye->bhxy", selected_queries, selected_keys).tanh()
        attended: Tensor = torch.einsum("bhxy,bhye->bhxe", scores, selected_values)
        mixed: Tensor = torch.einsum("bhxi,hiox->bhxo", attended, torch.view_as_complex(self.weight))
        output_spectrum: Tensor = torch.zeros_like(query_spectrum).index_copy(2, self.query_frequencies, mixed)
        outputs: Tensor = cast(Tensor, torch.fft.irfft(output_spectrum, n=self.query_length, dim=2))
        return outputs
