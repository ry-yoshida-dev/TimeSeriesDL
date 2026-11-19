from typing import override

import torch.nn.functional as functional
from torch import Tensor

from ..attention_mechanism import AttentionMechanism


class FullAttentionMechanism(AttentionMechanism):
    """
    Scaled dot-product attention over every key (Vaswani et al., 2017).

    Parameters
    ----------
    is_causal : bool
        Whether each query sees only the keys up to its own position.
    dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If dropout is not in [0, 1).
    """

    def __init__(self, is_causal: bool, dropout: float) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.is_causal: bool = is_causal
        self.dropout: float = dropout

    @override
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        outputs: Tensor = functional.scaled_dot_product_attention(
            queries, keys, values, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal
        )
        return outputs
