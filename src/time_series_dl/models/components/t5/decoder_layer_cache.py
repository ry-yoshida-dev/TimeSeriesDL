from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(eq=False)
class T5DecoderLayerCache:
    """
    Keys and values of one decoder block kept between the steps of incremental decoding: those of the decoder's own
    past positions, which grow by one position per step, and those of the encoder states, which are fixed.

    Attributes
    ----------
    self_key : Tensor
        Self-attention keys of the decoded positions with shape (batch, num_heads, length, key_value_size).
    self_value : Tensor
        Self-attention values with the same shape.
    cross_key : Tensor
        Cross-attention keys of the encoder states with shape (batch, num_heads, encoder_length, key_value_size).
    cross_value : Tensor
        Cross-attention values with the same shape.
    """

    self_key: Tensor
    self_value: Tensor
    cross_key: Tensor
    cross_value: Tensor

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """
        Add the self-attention keys and values of new positions.

        Parameters
        ----------
        key : Tensor
            Keys of the new positions with shape (batch, num_heads, new_length, key_value_size).
        value : Tensor
            Values with the same shape.

        Returns
        -------
        tuple[Tensor, Tensor]
            The keys and values of every decoded position.
        """
        self.self_key = torch.cat([self.self_key, key], dim=2)
        self.self_value = torch.cat([self.self_value, value], dim=2)
        return self.self_key, self.self_value
