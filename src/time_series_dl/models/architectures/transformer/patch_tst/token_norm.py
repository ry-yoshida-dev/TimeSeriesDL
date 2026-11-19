from typing import cast

from torch import Tensor, nn

from .token_normalization import TokenNormalization


class PatchTSTTokenNorm(nn.Module):
    """
    Batch or layer normalization of a batch of token sequences.

    Parameters
    ----------
    normalization : TokenNormalization
        The kind of normalization.
    model_size : int
        Size of each token vector.
    """

    def __init__(self, normalization: TokenNormalization, model_size: int) -> None:
        super().__init__()
        self.model_size: int = model_size
        self.norm: nn.BatchNorm1d | nn.LayerNorm = normalization.build(model_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Normalize the tokens.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Normalized tokens of the same shape.
        """
        normalized: Tensor = self.norm(tokens.reshape(-1, self.model_size))
        return normalized.reshape(tokens.shape)

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
