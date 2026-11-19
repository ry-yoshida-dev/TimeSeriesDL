from dataclasses import dataclass

import torch
from torch import Tensor

from ..masking import Masking


@dataclass(frozen=True, kw_only=True)
class GeometricMasking(Masking):
    """
    Masks of each channel made of alternating hidden and visible segments with geometrically distributed lengths,
    as in TST (Zerveas et al., 2021) and SimMTM (Dong et al., 2023).

    Each channel is a two-state Markov chain that starts hidden with probability ``ratio``; hidden segments last
    ``mean_length`` steps on average and visible ones ``mean_length * (1 - ratio) / ratio``, so a fraction ``ratio``
    of the values is hidden on average.

    Attributes
    ----------
    ratio : float
        Expected fraction of hidden values.
    mean_length : float
        Mean length of a hidden segment.

    Raises
    ------
    ValueError
        If ratio is not in (0, 1), mean_length is below 1, or visible segments would be shorter than one step on
        average.
    """

    ratio: float = 0.15
    mean_length: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {self.ratio}")
        if self.mean_length < 1.0:
            raise ValueError(f"mean_length must be at least 1, got {self.mean_length}")
        if self.visible_probability > 1.0:
            raise ValueError(
                f"ratio {self.ratio} is too high for mean_length {self.mean_length}: visible segments would be "
                + "shorter than one step on average"
            )

    @property
    def hidden_probability(self) -> float:
        """
        Get the probability that a hidden segment ends after each step.

        Returns
        -------
        float
            ``1 / mean_length``.
        """
        return 1.0 / self.mean_length

    @property
    def visible_probability(self) -> float:
        """
        Get the probability that a visible segment ends after each step.

        Returns
        -------
        float
            ``ratio / ((1 - ratio) * mean_length)``.
        """
        return self.ratio / ((1.0 - self.ratio) * self.mean_length)

    def sample(self, batch: int, length: int, channels: int, device: torch.device) -> Tensor:
        rows: int = batch * channels
        # Segment k of a row is hidden when the row starts hidden and k is even, or starts visible and k is odd;
        # ``length`` segments of at least one step always cover the sequence.
        is_start_hidden: Tensor = torch.rand(rows, 1, device=device) < self.ratio
        is_odd: Tensor = torch.arange(length, device=device) % 2 == 1
        is_segment_hidden: Tensor = is_start_hidden ^ is_odd
        hidden_lengths: Tensor = torch.empty(rows, length, device=device).geometric_(self.hidden_probability)
        visible_lengths: Tensor = torch.empty(rows, length, device=device).geometric_(self.visible_probability)
        ends: Tensor = torch.where(is_segment_hidden, hidden_lengths, visible_lengths).cumsum(dim=1)
        steps: Tensor = torch.arange(length, device=device, dtype=ends.dtype).expand(rows, length).contiguous()
        segments: Tensor = torch.searchsorted(ends, steps, right=True)
        is_hidden: Tensor = torch.gather(is_segment_hidden.expand(rows, length), 1, segments)
        return is_hidden.reshape(batch, channels, length).transpose(1, 2)
