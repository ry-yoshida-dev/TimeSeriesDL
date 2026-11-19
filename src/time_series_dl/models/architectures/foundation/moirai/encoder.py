from typing import cast

from torch import Tensor, nn

from .encoder_layer import MoiraiEncoderLayer
from .rotary_embedding import MoiraiRotaryEmbedding


class MoiraiEncoder(nn.Module):
    """
    Moirai's Transformer encoder: pre-normalized layers of any-variate attention sharing one rotary embedding, and a
    final root-mean-square normalization.

    Parameters
    ----------
    model_size : int
        Size of the hidden states.
    num_layers : int
        Number of layers.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward networks.
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
        num_layers: int,
        num_heads: int,
        feedforward_size: int,
        attention_dropout: float,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.rotary: MoiraiRotaryEmbedding = MoiraiRotaryEmbedding(model_size // num_heads)
        self.layers: nn.ModuleList = nn.ModuleList(
            MoiraiEncoderLayer(model_size, num_heads, feedforward_size, attention_dropout, dropout, epsilon)
            for _ in range(num_layers)
        )
        self.final_norm: nn.RMSNorm = nn.RMSNorm(model_size, eps=epsilon)

    def forward(self, hidden: Tensor, variate_ids: Tensor, time_ids: Tensor) -> Tensor:
        """
        Encode a batch of token sequences.

        Parameters
        ----------
        hidden : Tensor
            Token embeddings with shape (batch, tokens, model_size).
        variate_ids : Tensor
            Variate index of each token with shape (batch, tokens).
        time_ids : Tensor
            Time index of each token with shape (batch, tokens).

        Returns
        -------
        Tensor
            The token representations with the same shape.
        """
        for layer in self.layers:
            hidden = layer(hidden, variate_ids, time_ids, self.rotary)
        outputs: Tensor = self.final_norm(hidden)
        return outputs

    def __call__(self, hidden: Tensor, variate_ids: Tensor, time_ids: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(hidden, variate_ids, time_ids))
