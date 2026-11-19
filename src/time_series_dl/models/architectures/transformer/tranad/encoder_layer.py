from typing import cast

from torch import Tensor, nn


class TranADEncoderLayer(nn.Module):
    """
    Self-attention and a leaky-ReLU feed-forward network, each with a residual connection and no layer
    normalization, as in the official TranAD code.

    Parameters
    ----------
    model_size : int
        Size of each token.
    num_heads : int
        Number of attention heads; must divide model_size.
    feedforward_size : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability.
    """

    def __init__(self, model_size: int, num_heads: int, feedforward_size: int, dropout: float) -> None:
        super().__init__()
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            model_size, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(model_size, feedforward_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, model_size),
        )
        self.attention_dropout: nn.Dropout = nn.Dropout(dropout)
        self.feedforward_dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Encode a batch of token sequences.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Encoded tokens with the same shape.
        """
        attended: Tensor = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        tokens = tokens + self.attention_dropout(attended)
        encoded: Tensor = tokens + self.feedforward_dropout(self.feedforward(tokens))
        return encoded

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
