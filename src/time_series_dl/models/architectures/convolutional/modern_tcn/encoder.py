from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .block import ModernTCNBlock

if TYPE_CHECKING:
    from .parameters import ModernTCNParameters


class ModernTCNEncoder(nn.Module):
    """
    ModernTCN backbone: each variable is cut into patches embedded by a strided convolution, and stages of ModernTCN
    blocks, separated by strided downsampling convolutions, mix time, features and variables.

    As in the official code, the series is extended by repeating its last value by ``patch_size - patch_stride``
    steps before patching (more when the series is shorter than one stride, so there is at least one patch), and a
    stage input whose number of patches is not a multiple of the downsampling ratio is extended by repeating its
    last patches.

    Parameters
    ----------
    parameters : ModernTCNParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of variables of the input.

    Raises
    ------
    ValueError
        If input_length or input_channels is not positive.
    """

    def __init__(self, parameters: ModernTCNParameters, input_length: int, input_channels: int) -> None:
        if input_length <= 0 or input_channels <= 0:
            raise ValueError(
                f"input_length and input_channels must be positive, got {input_length} and {input_channels}"
            )
        super().__init__()
        self.downsample_ratio: int = parameters.downsample_ratio
        self.stem_padding: int = parameters.patch_size - min(parameters.patch_stride, input_length)
        sizes: list[int] = [stage.model_size for stage in parameters.stages]
        self.downsampling: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, sizes[0], parameters.patch_size, stride=parameters.patch_stride),
                    nn.BatchNorm1d(sizes[0]),
                ),
                *(
                    nn.Sequential(
                        nn.BatchNorm1d(sizes[position - 1]),
                        nn.Conv1d(
                            sizes[position - 1],
                            sizes[position],
                            parameters.downsample_ratio,
                            stride=parameters.downsample_ratio,
                        ),
                    )
                    for position in range(1, len(sizes))
                ),
            ]
        )
        self.stages: nn.ModuleList = nn.ModuleList(
            nn.Sequential(
                *(
                    ModernTCNBlock(stage, input_channels, parameters.feedforward_ratio, parameters.dropout)
                    for _ in range(stage.num_blocks)
                )
            )
            for stage in parameters.stages
        )
        num_patches: int = (input_length + self.stem_padding - parameters.patch_size) // parameters.patch_stride + 1
        for _ in range(1, len(sizes)):
            num_patches = math.ceil(num_patches / parameters.downsample_ratio)
        self.output_length: int = num_patches
        self.output_size: int = sizes[-1]

    def forward(self, values: Tensor) -> Tensor:
        """
        Encode a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, input_channels).

        Returns
        -------
        Tensor
            Features with shape (batch, input_channels, output_size, output_length).
        """
        batch_size: int = values.shape[0]
        num_variables: int = values.shape[2]
        features: Tensor = values.transpose(1, 2).reshape(batch_size * num_variables, 1, -1)
        features = torch.cat([features, features[:, :, -1:].expand(-1, -1, self.stem_padding)], dim=-1)
        for position, (downsampling, stage) in enumerate(zip(self.downsampling, self.stages, strict=True)):
            if position > 0:
                features = self._extend(features.flatten(end_dim=1))
            embedded: Tensor = downsampling(features)
            features = stage(embedded.reshape(batch_size, num_variables, embedded.shape[1], embedded.shape[2]))
        return features

    def _extend(self, features: Tensor) -> Tensor:
        num_patches: int = features.shape[-1]
        padding: int = -num_patches % self.downsample_ratio
        indices: Tensor = torch.arange(num_patches - padding, num_patches, device=features.device).clamp(min=0)
        return torch.cat([features, features.index_select(-1, indices)], dim=-1)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
