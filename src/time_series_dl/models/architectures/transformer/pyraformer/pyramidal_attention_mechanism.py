from typing import override

import torch
import torch.nn.functional as functional
from torch import Tensor

from ....components.attention.attention_mechanism import AttentionMechanism


class PyraformerPyramidalAttentionMechanism(AttentionMechanism):
    """
    Scaled dot-product attention restricted by the pyramidal mask, computed densely.

    Parameters
    ----------
    mask : Tensor
        Boolean mask with shape (num_nodes, num_nodes), True where a query may attend to a key.
    dropout : float
        Dropout probability of the attention weights.

    Raises
    ------
    ValueError
        If the mask is not a square boolean matrix or dropout is not in [0, 1).
    """

    def __init__(self, mask: Tensor, dropout: float) -> None:
        if mask.dtype != torch.bool or mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
            raise ValueError(f"mask must be a square boolean matrix, got {mask.dtype} with shape {tuple(mask.shape)}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.dropout: float = dropout
        self.mask: Tensor
        self.register_buffer("mask", mask, persistent=False)

    @override
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        outputs: Tensor = functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=self.mask, dropout_p=self.dropout if self.training else 0.0
        )
        return outputs
