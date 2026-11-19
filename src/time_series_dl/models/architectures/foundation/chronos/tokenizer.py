from typing import Final

import torch
from torch import Tensor, nn

PAD_TOKEN: Final[int] = 0
EOS_TOKEN: Final[int] = 1
NUM_SPECIAL_TOKENS: Final[int] = 2
_BOUNDARY_LIMIT: Final[float] = 1e20


class ChronosTokenizer(nn.Module):
    """
    Mean scaling and uniform quantization of Chronos (``MeanScaleUniformBins``): each series is divided by the mean
    absolute value of its observed context, and every scaled value becomes the token of the nearest of
    ``num_tokens - 3`` evenly spaced bin centers in [low_limit, high_limit], after the special tokens PAD (0) and
    EOS (1). Missing values become PAD.

    Parameters
    ----------
    num_tokens : int
        Size of the vocabulary, including the special tokens.
    low_limit : float
        Center of the lowest bin, in units of the scale.
    high_limit : float
        Center of the highest bin, in units of the scale.
    """

    def __init__(self, num_tokens: int, low_limit: float, high_limit: float) -> None:
        super().__init__()
        self.num_tokens: int = num_tokens
        centers: Tensor = torch.linspace(low_limit, high_limit, num_tokens - NUM_SPECIAL_TOKENS - 1)
        boundaries: Tensor = torch.cat(
            [
                torch.tensor([-_BOUNDARY_LIMIT]),
                (centers[1:] + centers[:-1]) / 2,
                torch.tensor([_BOUNDARY_LIMIT]),
            ]
        )
        self.centers: Tensor
        self.register_buffer("centers", centers, persistent=False)
        self.boundaries: Tensor
        self.register_buffer("boundaries", boundaries, persistent=False)

    @staticmethod
    def compute_scale(context: Tensor) -> Tensor:
        """
        Compute the mean absolute value of the observed values of each series.

        Parameters
        ----------
        context : Tensor
            Values with shape (batch, length); NaN marks missing values.

        Returns
        -------
        Tensor
            The scale with shape (batch,); 1 where it is not positive (all zero or nothing observed).
        """
        is_observed: Tensor = ~torch.isnan(context)
        total: Tensor = torch.where(is_observed, context.abs(), 0.0).sum(dim=-1)
        scale: Tensor = total / is_observed.sum(dim=-1)
        return torch.where(scale > 0, scale, 1.0)

    def encode(self, values: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        """
        Turn values into tokens.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length); NaN marks missing values.
        scale : Tensor
            Scale of each series with shape (batch,).

        Returns
        -------
        tuple[Tensor, Tensor]
            The token ids (int64) and the boolean observation mask, each with shape (batch, length).
        """
        values = values.to(torch.float32)
        is_observed: Tensor = ~torch.isnan(values)
        scaled: Tensor = values / scale.unsqueeze(-1)
        tokens: Tensor = torch.bucketize(scaled.contiguous(), self.boundaries, right=True) + NUM_SPECIAL_TOKENS
        tokens = tokens.clamp(0, self.num_tokens - 1)
        return torch.where(is_observed, tokens, PAD_TOKEN), is_observed

    def decode(self, tokens: Tensor, scale: Tensor) -> Tensor:
        """
        Turn tokens into the bin centers they stand for, in the scale of the series.

        Parameters
        ----------
        tokens : Tensor
            Token ids with shape (batch, ...).
        scale : Tensor
            Scale of each series with shape (batch,).

        Returns
        -------
        Tensor
            The values with the shape of ``tokens``; special tokens map to the lowest center.
        """
        indices: Tensor = (tokens - NUM_SPECIAL_TOKENS - 1).clamp(0, self.centers.shape[0] - 1)
        return self.centers[indices] * scale.reshape(-1, *([1] * (tokens.dim() - 1)))
