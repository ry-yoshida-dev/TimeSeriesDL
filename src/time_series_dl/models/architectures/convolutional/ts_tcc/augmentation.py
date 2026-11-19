from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class TSTCCAugmentation:
    """
    Weak and strong augmentations of TS-TCC, drawn anew on every call.

    As in the official code, the weak augmentation scales every time step of a sequence by a factor drawn from
    N(2, scale_ratio^2) and shared by its channels, and the strong augmentation permutes random segments of the time
    axis and adds Gaussian noise.

    Attributes
    ----------
    scale_ratio : float
        Standard deviation of the scaling factors of the weak augmentation.
    jitter_ratio : float
        Standard deviation of the noise of the strong augmentation.
    max_segments : int
        Bound of the number of permuted segments, drawn uniformly from ``[1, max_segments)``.

    Raises
    ------
    ValueError
        If a ratio is negative or max_segments is below 2.
    """

    scale_ratio: float = 1.1
    jitter_ratio: float = 0.8
    max_segments: int = 8

    def __post_init__(self) -> None:
        if self.scale_ratio < 0.0:
            raise ValueError(f"scale_ratio must not be negative, got {self.scale_ratio}")
        if self.jitter_ratio < 0.0:
            raise ValueError(f"jitter_ratio must not be negative, got {self.jitter_ratio}")
        if self.max_segments < 2:
            raise ValueError(f"max_segments must be at least 2, got {self.max_segments}")

    def weak(self, values: Tensor) -> Tensor:
        """
        Apply the weak augmentation.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Scaled values of the same shape.
        """
        return self.scale(values)

    def strong(self, values: Tensor) -> Tensor:
        """
        Apply the strong augmentation.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Permuted and jittered values of the same shape.
        """
        return self.jitter(self.permute(values))

    def scale(self, values: Tensor) -> Tensor:
        """
        Multiply every time step by a random factor shared by its channels.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Scaled values of the same shape.
        """
        factors: Tensor = 2.0 + self.scale_ratio * torch.randn(
            values.shape[0], values.shape[1], 1, device=values.device
        )
        return values * factors

    def jitter(self, values: Tensor) -> Tensor:
        """
        Add Gaussian noise to every value.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Noisy values of the same shape.
        """
        return values + self.jitter_ratio * torch.randn_like(values)

    def permute(self, values: Tensor) -> Tensor:
        """
        Split each sequence into a random number of random contiguous segments and shuffle the segments.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Permuted values of the same shape; every channel of a sequence is permuted alike.
        """
        batch: int = values.shape[0]
        length: int = values.shape[1]
        device: torch.device = values.device
        num_segments: Tensor = torch.randint(1, self.max_segments, (batch, 1), device=device).clamp(max=length)
        # The first num_segments - 1 of a random order of the gaps between steps split the sequence.
        gap_ranks: Tensor = torch.rand(batch, length - 1, device=device).argsort(dim=1).argsort(dim=1)
        is_split: Tensor = gap_ranks < num_segments - 1
        segments: Tensor = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.long, device=device), is_split.long().cumsum(dim=1)], dim=1
        )
        segment_ranks: Tensor = torch.rand(batch, self.max_segments, device=device).argsort(dim=1).argsort(dim=1)
        steps: Tensor = torch.arange(length, device=device)
        order: Tensor = (segment_ranks.gather(1, segments) * length + steps).argsort(dim=1)
        return values.gather(1, order.unsqueeze(-1).expand(-1, -1, values.shape[2]))
