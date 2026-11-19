from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .encoder_layer import PatchTSTEncoderLayer

if TYPE_CHECKING:
    from .parameters import PatchTSTParameters


class PatchTSTEncoder(nn.Module):
    """
    Channel-independent backbone: each channel is cut into patches that a shared Transformer encodes on its own.

    The series is padded at the end with ``stride`` copies of its last value before patching, as in the paper, so
    there are ``(input_length + stride - patch_length) // stride + 1`` patches.

    Parameters
    ----------
    parameters : PatchTSTParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.

    Raises
    ------
    ValueError
        If input_length is not positive or the padded series is shorter than one patch.
    """

    def __init__(self, parameters: PatchTSTParameters, input_length: int) -> None:
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if parameters.patch_length > input_length + parameters.stride:
            raise ValueError(
                f"patch_length {parameters.patch_length} must not exceed input_length {input_length} + stride "
                + f"{parameters.stride}"
            )
        super().__init__()
        self.patch_length: int = parameters.patch_length
        self.stride: int = parameters.stride
        self.model_size: int = parameters.model_size
        self.num_patches: int = (input_length + parameters.stride - parameters.patch_length) // parameters.stride + 1
        self.embedding: nn.Linear = nn.Linear(parameters.patch_length, parameters.model_size)
        self.position: nn.Parameter = nn.Parameter(
            torch.empty(self.num_patches, parameters.model_size).uniform_(-0.02, 0.02)
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.layers: nn.Sequential = nn.Sequential(
            *(PatchTSTEncoderLayer(parameters) for _ in range(parameters.num_layers))
        )

    @property
    def feature_size(self) -> int:
        """
        Get the size of the flattened features of one channel.

        Returns
        -------
        int
            ``num_patches * model_size``.
        """
        return self.num_patches * self.model_size

    def forward(self, values: Tensor) -> Tensor:
        """
        Encode every channel of a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        Tensor
            Patch features with shape (batch, channels, num_patches, model_size).
        """
        batch: int = values.shape[0]
        channels: int = values.shape[2]
        series: Tensor = values.transpose(1, 2)
        padded: Tensor = torch.cat([series, series[..., -1:].expand(-1, -1, self.stride)], dim=-1)
        patches: Tensor = padded.unfold(-1, self.patch_length, self.stride)
        embedded: Tensor = self.embedding(patches)
        tokens: Tensor = embedded.reshape(batch * channels, self.num_patches, self.model_size) + self.position
        encoded: Tensor = self.layers(self.dropout(tokens))
        return encoded.reshape(batch, channels, self.num_patches, self.model_size)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
