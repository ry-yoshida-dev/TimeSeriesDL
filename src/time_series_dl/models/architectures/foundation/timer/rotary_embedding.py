from typing import cast

import torch
from torch import Tensor, nn


class TimerRotaryEmbedding(nn.Module):
    """
    Rotary position embedding of the queries and keys of Timer: each pair of features (``i``, ``i + size / 2``) is
    rotated by the angle ``position / base ** (2 i / size)``.

    Parameters
    ----------
    size : int
        Size of each attention head; must be even.
    base : float
        Base of the rotation frequencies.

    Raises
    ------
    ValueError
        If size is not even.
    """

    def __init__(self, size: int, base: float) -> None:
        if size % 2 != 0:
            raise ValueError(f"the head size must be even for rotary embeddings, got {size}")
        super().__init__()
        self.inverse_frequency: Tensor
        self.register_buffer(
            "inverse_frequency",
            1.0 / base ** (torch.arange(0, size, 2, dtype=torch.int64).to(torch.float32) / size),
            persistent=False,
        )

    def forward(self, values: Tensor) -> Tensor:
        """
        Rotate queries or keys by their positions.

        Parameters
        ----------
        values : Tensor
            Queries or keys with shape (batch, num_heads, length, size), at positions 0 to length - 1.

        Returns
        -------
        Tensor
            The rotated values.
        """
        positions: Tensor = torch.arange(values.shape[-2], device=values.device, dtype=torch.float32)
        frequencies: Tensor = torch.outer(positions, self.inverse_frequency)
        angles: Tensor = torch.cat([frequencies, frequencies], dim=-1)
        half: int = values.shape[-1] // 2
        rotated: Tensor = torch.cat([-values[..., half:], values[..., :half]], dim=-1)
        return values * angles.cos().to(values.dtype) + rotated * angles.sin().to(values.dtype)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
