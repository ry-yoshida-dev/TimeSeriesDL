from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.attention.multi_head_attention import MultiHeadAttention
from .pyramidal_attention_mechanism import PyraformerPyramidalAttentionMechanism


class PyraformerEncoderLayer(nn.Module):
    """
    Pyramidal self-attention and a GELU feed-forward network, each with dropout, a residual connection and a following
    layer normalization, as in the official code with ``normalize_before`` disabled.

    Parameters
    ----------
    mask : Tensor
        Boolean pyramidal mask with shape (num_nodes, num_nodes), True where a query may attend to a key.
    model_size : int
        Size of each node token.
    num_heads : int
        Number of attention heads.
    feedforward_size : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability of the attention weights and of the sublayer outputs.
    """

    def __init__(self, mask: Tensor, model_size: int, num_heads: int, feedforward_size: int, dropout: float) -> None:
        super().__init__()
        self.attention: MultiHeadAttention = MultiHeadAttention(
            PyraformerPyramidalAttentionMechanism(mask, dropout), model_size, num_heads
        )
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(model_size, eps=1e-6)
        self.hidden: nn.Linear = nn.Linear(model_size, feedforward_size)
        self.output: nn.Linear = nn.Linear(feedforward_size, model_size)
        self.feed_forward_norm: nn.LayerNorm = nn.LayerNorm(model_size, eps=1e-6)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor) -> Tensor:
        """
        Apply the layer.

        Parameters
        ----------
        nodes : Tensor
            Node tokens with shape (batch, num_nodes, model_size).

        Returns
        -------
        Tensor
            Node tokens of the same shape.
        """
        attended: Tensor = self.attention_norm(nodes + self.dropout(self.attention(nodes, nodes, nodes)))
        transformed: Tensor = self.output(self.dropout(functional.gelu(self.hidden(attended))))
        outputs: Tensor = self.feed_forward_norm(attended + self.dropout(transformed))
        return outputs

    def __call__(self, nodes: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(nodes))
