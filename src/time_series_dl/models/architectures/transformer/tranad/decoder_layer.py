from typing import cast

from torch import Tensor, nn


class TranADDecoderLayer(nn.Module):
    """
    Self-attention over the target tokens, attention to the encoder memory and a leaky-ReLU feed-forward network,
    each with a residual connection and no layer normalization, as in the official TranAD code.

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
        self.self_attention: nn.MultiheadAttention = nn.MultiheadAttention(
            model_size, num_heads, dropout=dropout, batch_first=True
        )
        self.memory_attention: nn.MultiheadAttention = nn.MultiheadAttention(
            model_size, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(model_size, feedforward_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, model_size),
        )
        self.self_attention_dropout: nn.Dropout = nn.Dropout(dropout)
        self.memory_attention_dropout: nn.Dropout = nn.Dropout(dropout)
        self.feedforward_dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        """
        Decode a batch of target sequences.

        Parameters
        ----------
        target : Tensor
            Target tokens with shape (batch, length, model_size).
        memory : Tensor
            Encoder memory with shape (batch, memory_length, model_size).

        Returns
        -------
        Tensor
            Decoded tokens with the shape of the target.
        """
        attended: Tensor = self.self_attention(target, target, target, need_weights=False)[0]
        target = target + self.self_attention_dropout(attended)
        recalled: Tensor = self.memory_attention(target, memory, memory, need_weights=False)[0]
        target = target + self.memory_attention_dropout(recalled)
        decoded: Tensor = target + self.feedforward_dropout(self.feedforward(target))
        return decoded

    def __call__(self, target: Tensor, memory: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(target, memory))
