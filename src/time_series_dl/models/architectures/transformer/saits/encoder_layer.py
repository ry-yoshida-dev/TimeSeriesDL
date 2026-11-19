from typing import cast

from torch import Tensor, nn

from .diagonally_masked_attention import SAITSDiagonallyMaskedAttention


class SAITSEncoderLayer(nn.Module):
    """
    Diagonally masked self-attention and a position-wise feed-forward network, each with a residual connection
    followed by layer normalization.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    num_heads : int
        Number of attention heads.
    key_size : int
        Size of the queries and keys of each head.
    value_size : int
        Size of the values of each head.
    feedforward_size : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability after the attention and the feed-forward network.
    attention_dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If a size is not positive or a dropout is not in [0, 1).
    """

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        key_size: int,
        value_size: int,
        feedforward_size: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        if feedforward_size <= 0:
            raise ValueError(f"feedforward_size must be positive, got {feedforward_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.attention: SAITSDiagonallyMaskedAttention = SAITSDiagonallyMaskedAttention(
            model_size, num_heads, key_size, value_size, attention_dropout
        )
        self.attention_dropout: nn.Dropout = nn.Dropout(dropout)
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(model_size, eps=1e-6)
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(model_size, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, model_size),
            nn.Dropout(dropout),
        )
        self.feedforward_norm: nn.LayerNorm = nn.LayerNorm(model_size, eps=1e-6)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Transform a sequence of tokens.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        tuple[Tensor, Tensor]
            The output tokens of the input's shape, and the attention weights with shape
            (batch, num_heads, length, length).
        """
        attended, weights = self.attention(tokens)
        hidden: Tensor = self.attention_norm(tokens + self.attention_dropout(attended))
        outputs: Tensor = self.feedforward_norm(hidden + self.feedforward(hidden))
        return outputs, weights

    def __call__(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(tokens))
