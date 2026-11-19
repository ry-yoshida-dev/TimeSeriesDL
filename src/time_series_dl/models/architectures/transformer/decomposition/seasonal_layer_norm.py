from typing import cast

from torch import Tensor, nn


class SeasonalLayerNorm(nn.Module):
    """
    Layer normalization of each token followed by subtraction of the temporal mean, so the output stays seasonal.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    """

    def __init__(self, model_size: int) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(model_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Normalize a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Normalized tokens with zero mean over time.
        """
        normalized: Tensor = self.norm(tokens)
        return normalized - normalized.mean(dim=1, keepdim=True)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
