from typing import cast

from torch import Tensor, nn

from .attention import MoiraiAttention
from .feed_forward import MoiraiFeedForward
from .rotary_embedding import MoiraiRotaryEmbedding


class MoiraiEncoderLayer(nn.Module):
    """
    Pre-normalized Moirai encoder layer: any-variate attention and a gated feed-forward network, each added to its
    input.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward network.
    attention_dropout : float
        Dropout probability of the attention weights.
    dropout : float
        Dropout probability of the residual branches.
    epsilon : float
        Epsilon of the root-mean-square normalizations.
    """

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        feedforward_size: int,
        attention_dropout: float,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.attention_norm: nn.RMSNorm = nn.RMSNorm(model_size, eps=epsilon)
        self.attention: MoiraiAttention = MoiraiAttention(model_size, num_heads, attention_dropout, epsilon)
        self.feed_forward_norm: nn.RMSNorm = nn.RMSNorm(model_size, eps=epsilon)
        self.feed_forward: MoiraiFeedForward = MoiraiFeedForward(model_size, feedforward_size, dropout)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, variate_ids: Tensor, time_ids: Tensor, rotary: MoiraiRotaryEmbedding) -> Tensor:
        """
        Transform a batch of token sequences.

        Parameters
        ----------
        hidden : Tensor
            Token representations with shape (batch, tokens, model_size).
        variate_ids : Tensor
            Variate index of each token with shape (batch, tokens).
        time_ids : Tensor
            Time index of each token with shape (batch, tokens).
        rotary : MoiraiRotaryEmbedding
            The rotary embedding shared by every layer.

        Returns
        -------
        Tensor
            The new representations.
        """
        attended: Tensor = self.dropout(self.attention(self.attention_norm(hidden), variate_ids, time_ids, rotary))
        hidden = hidden + attended
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))

    def __call__(self, hidden: Tensor, variate_ids: Tensor, time_ids: Tensor, rotary: MoiraiRotaryEmbedding) -> Tensor:
        return cast(Tensor, super().__call__(hidden, variate_ids, time_ids, rotary))
