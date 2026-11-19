from typing import cast

from torch import Tensor, nn

from ....components.layers import Activation
from ....components.state_space import MambaStateSpaceLayer


class SMambaEncoderLayer(nn.Module):
    """
    Encoder layer of S-Mamba: two Mamba blocks read the variate tokens in opposite orders, then a position-wise
    feed-forward network, each with a residual connection and layer normalization.

    The Mamba blocks are not expanded and use a convolution of size 2, as in the paper.

    Parameters
    ----------
    model_size : int
        Size of each variate token.
    state_size : int
        Size of the state of each channel of the Mamba blocks.
    feedforward_size : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability of the residual branches.
    activation : Activation
        Activation of the feed-forward network.
    """

    def __init__(
        self, model_size: int, state_size: int, feedforward_size: int, dropout: float, activation: Activation
    ) -> None:
        super().__init__()
        self.forward_mamba: MambaStateSpaceLayer = MambaStateSpaceLayer(model_size, state_size, 1, 2)
        self.backward_mamba: MambaStateSpaceLayer = MambaStateSpaceLayer(model_size, state_size, 1, 2)
        self.mixing_normalization: nn.LayerNorm = nn.LayerNorm(model_size)
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(model_size, feedforward_size),
            activation.build(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, model_size),
            nn.Dropout(dropout),
        )
        self.feedforward_normalization: nn.LayerNorm = nn.LayerNorm(model_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Mix the variate tokens.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, channels, model_size).

        Returns
        -------
        Tensor
            Tokens with the shape of ``tokens``.
        """
        mixed: Tensor = self.forward_mamba(tokens) + self.backward_mamba(tokens.flip(1)).flip(1)
        normalized: Tensor = self.mixing_normalization(tokens + self.dropout(mixed))
        outputs: Tensor = self.feedforward_normalization(normalized + self.feedforward(normalized))
        return outputs

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
