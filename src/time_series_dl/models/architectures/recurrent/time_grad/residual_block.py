import math
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

_NEGATIVE_SLOPE: float = 0.4


class TimeGradResidualBlock(nn.Module):
    """
    Gated, dilated convolution block of the TimeGrad denoiser, which convolves along the target dimensions of one
    time step (as DiffWave convolves along time).

    The diffusion step embedding is added to the input, the dilated convolution and the projected conditioning are
    summed, and a tanh-sigmoid gate yields the residual and the skip outputs.

    Parameters
    ----------
    channels : int
        Number of residual channels.
    dilation : int
        Dilation of the convolution of kernel size 3.
    step_size : int
        Size of the diffusion step embedding.
    """

    def __init__(self, channels: int, dilation: int, step_size: int) -> None:
        super().__init__()
        self.step_projection: nn.Linear = nn.Linear(step_size, channels)
        self.dilated_convolution: nn.Conv1d = nn.Conv1d(channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        self.conditioning_projection: nn.Conv1d = nn.Conv1d(1, 2 * channels, 1)
        self.output_projection: nn.Conv1d = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, values: Tensor, conditioning: Tensor, step_embedding: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the block.

        Parameters
        ----------
        values : Tensor
            Hidden values with shape (batch, channels, target_dimensions).
        conditioning : Tensor
            Upsampled conditioning with shape (batch, 1, target_dimensions).
        step_embedding : Tensor
            Diffusion step embedding with shape (batch, step_size).

        Returns
        -------
        tuple[Tensor, Tensor]
            The residual output and the skip output, each with the shape of ``values``.
        """
        hidden: Tensor = values + self.step_projection(step_embedding).unsqueeze(-1)
        hidden = self.dilated_convolution(hidden) + self.conditioning_projection(conditioning)
        gate: Tensor
        content: Tensor
        gate, content = hidden.chunk(2, dim=1)
        hidden = functional.leaky_relu(
            self.output_projection(torch.sigmoid(gate) * torch.tanh(content)), _NEGATIVE_SLOPE
        )
        residual: Tensor
        skip: Tensor
        residual, skip = hidden.chunk(2, dim=1)
        return (values + residual) / math.sqrt(2.0), skip

    def __call__(self, values: Tensor, conditioning: Tensor, step_embedding: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values, conditioning, step_embedding))
