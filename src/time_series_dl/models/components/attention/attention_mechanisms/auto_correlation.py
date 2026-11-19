import math
from typing import cast, override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..attention_mechanism import AttentionMechanism


class AutoCorrelationAttentionMechanism(AttentionMechanism):
    """
    Auto-correlation of Autoformer (Wu et al., NeurIPS 2021): period-based dependencies by time-delay aggregation.

    The correlation of queries and keys at every delay is computed with the fast Fourier transform and averaged over
    heads and channels. The values rolled by the ``int(factor * ln L)`` delays of highest correlation are summed with
    the softmax of those correlations as weights. Delays are chosen per sample, as in the paper's inference path.
    Keys and values are truncated, or padded with zeros, to the length of the queries.

    Parameters
    ----------
    factor : int
        Factor c of the number of delays ``c * ln L``.

    Raises
    ------
    ValueError
        If factor is not positive.
    """

    def __init__(self, factor: int) -> None:
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        super().__init__()
        self.factor: int = factor

    def count_delays(self, length: int) -> int:
        """
        Get the number of aggregated delays.

        Parameters
        ----------
        length : int
            Number of time steps.

        Returns
        -------
        int
            ``int(factor * ln length)``, clipped to [1, length].
        """
        return max(1, min(int(self.factor * math.log(length)), length))

    def compute_delays(self, queries: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        """
        Select the delays of highest auto-correlation and their aggregation weights.

        Parameters
        ----------
        queries : Tensor
            Queries with shape (batch, heads, length, head_size).
        keys : Tensor
            Keys with shape (batch, heads, length, head_size).

        Returns
        -------
        tuple[Tensor, Tensor]
            The delays with shape (batch, count_delays(length)), and their weights of the same shape, summing to one.
        """
        length: int = queries.shape[2]
        query_spectrum: Tensor = cast(Tensor, torch.fft.rfft(queries, dim=2))
        key_spectrum: Tensor = cast(Tensor, torch.fft.rfft(keys, dim=2))
        spectrum: Tensor = query_spectrum * key_spectrum.conj()
        correlation: Tensor = cast(Tensor, torch.fft.irfft(spectrum, n=length, dim=2)).mean(dim=(1, 3))
        top: torch.return_types.topk = correlation.topk(self.count_delays(length), dim=-1)
        return top.indices, top.values.softmax(dim=-1)

    @override
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        length: int = queries.shape[2]
        aligned_keys: Tensor = self._align(keys, length)
        aligned_values: Tensor = self._align(values, length)
        delays: Tensor
        weights: Tensor
        delays, weights = self.compute_delays(queries, aligned_keys)
        doubled: Tensor = torch.cat([aligned_values, aligned_values], dim=2)
        positions: Tensor = delays.unsqueeze(-1) + torch.arange(length, device=queries.device)
        batch_index: Tensor = torch.arange(queries.shape[0], device=queries.device).view(-1, 1, 1)
        # Advanced indices on dimensions 0 and 2 move to the front: (batch, delays, length, heads, head_size).
        rolled: Tensor = doubled[batch_index, :, positions, :]
        aggregated: Tensor = (rolled * weights.view(*weights.shape, 1, 1, 1)).sum(dim=1)
        return aggregated.permute(0, 2, 1, 3)

    @staticmethod
    def _align(sequence: Tensor, length: int) -> Tensor:
        if sequence.shape[2] >= length:
            return sequence[:, :, :length, :]
        return functional.pad(sequence, (0, 0, 0, length - sequence.shape[2]))
