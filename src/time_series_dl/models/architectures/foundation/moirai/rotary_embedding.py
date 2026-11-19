import torch
from torch import Tensor


class MoiraiRotaryEmbedding:
    """
    Rotary position embedding of Moirai over the time index of each token, applied to the first half of every
    query and key head (the other half is left unrotated); pairs of adjacent features are rotated together.

    Parameters
    ----------
    head_size : int
        Size of each attention head.
    base : float
        Base of the geometric frequencies.

    Raises
    ------
    ValueError
        If half the head size is not even.
    """

    def __init__(self, head_size: int, base: float = 10000.0) -> None:
        width: int = head_size // 2
        if width % 2 != 0:
            raise ValueError(f"half the head size must be even, got {width}")
        self.width: int = width
        self.frequencies: Tensor = 1.0 / torch.pow(base, torch.arange(0, width, 2, dtype=torch.float32) / width)

    def apply(self, values: Tensor, positions: Tensor) -> Tensor:
        """
        Rotate queries or keys by the time index of their tokens.

        Parameters
        ----------
        values : Tensor
            Queries or keys with shape (batch, heads, tokens, head_size).
        positions : Tensor
            Integer time indices with shape (batch, tokens).

        Returns
        -------
        Tensor
            The rotated values, with the same shape.
        """
        frequencies: Tensor = self.frequencies.to(values.device)
        angles: Tensor = positions.to(torch.float32)[..., None] * frequencies
        angles = angles.repeat_interleave(2, dim=-1)[:, None]
        rotated: Tensor = values[..., : self.width]
        kept: Tensor = values[..., self.width :]
        even: Tensor = rotated[..., 0::2]
        odd: Tensor = rotated[..., 1::2]
        turned: Tensor = torch.stack([-odd, even], dim=-1).flatten(start_dim=-2)
        result: Tensor = angles.cos().to(values.dtype) * rotated + angles.sin().to(values.dtype) * turned
        return torch.cat([result, kept], dim=-1)
