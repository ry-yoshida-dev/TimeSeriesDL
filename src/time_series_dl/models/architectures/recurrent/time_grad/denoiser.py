from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.diffusion import DiffusionStepEmbedding
from .residual_block import TimeGradResidualBlock

if TYPE_CHECKING:
    from .parameters import TimeGradParameters

_NEGATIVE_SLOPE: float = 0.4


class TimeGradDenoiser(nn.Module):
    """
    Noise predictor of TimeGrad (the paper's epsilon_theta): predicts the noise of a noisy multivariate step from the
    diffusion step and the recurrent conditioning, with residual blocks that convolve along the target dimensions.

    Parameters
    ----------
    parameters : TimeGradParameters
        The model configuration.
    target_dimensions : int
        Number of target channels of one step.
    """

    def __init__(self, parameters: TimeGradParameters, target_dimensions: int) -> None:
        super().__init__()
        channels: int = parameters.residual_channels
        upsampled: int = max(target_dimensions // 2, 1)
        self.input_projection: nn.Conv1d = nn.Conv1d(1, channels, 1)
        self.step_embedding: DiffusionStepEmbedding = DiffusionStepEmbedding(
            parameters.step_feature_size, parameters.step_projection_size
        )
        self.conditioning_hidden: nn.Linear = nn.Linear(parameters.conditioning_size, upsampled)
        self.conditioning_output: nn.Linear = nn.Linear(upsampled, target_dimensions)
        self.blocks: nn.ModuleList = nn.ModuleList(
            TimeGradResidualBlock(
                channels, 2 ** (index % parameters.dilation_cycle_length), parameters.step_projection_size
            )
            for index in range(parameters.residual_layers)
        )
        self.skip_projection: nn.Conv1d = nn.Conv1d(channels, channels, 3, padding=1)
        self.output_projection: nn.Conv1d = nn.Conv1d(channels, 1, 3, padding=1)
        _ = nn.init.zeros_(self.output_projection.weight)

    def forward(self, noisy: Tensor, steps: Tensor, conditioning: Tensor) -> Tensor:
        """
        Predict the noise of noisy steps.

        Parameters
        ----------
        noisy : Tensor
            Noisy target values with shape (batch, target_dimensions).
        steps : Tensor
            Zero-based diffusion steps with shape (batch,).
        conditioning : Tensor
            Projected hidden states with shape (batch, conditioning_size).

        Returns
        -------
        Tensor
            The predicted noise with shape (batch, target_dimensions).
        """
        hidden: Tensor = functional.leaky_relu(self.input_projection(noisy.unsqueeze(1)), _NEGATIVE_SLOPE)
        step_embedding: Tensor = self.step_embedding(steps)
        upsampled: Tensor = self.conditioning_output(
            functional.leaky_relu(self.conditioning_hidden(conditioning), _NEGATIVE_SLOPE)
        ).unsqueeze(1)
        skips: Tensor = hidden.new_zeros(hidden.shape)
        for block in self.blocks:
            skip: Tensor
            hidden, skip = cast(TimeGradResidualBlock, block)(hidden, upsampled, step_embedding)
            skips = skips + skip
        output: Tensor = functional.leaky_relu(
            self.skip_projection(skips / math.sqrt(len(self.blocks))), _NEGATIVE_SLOPE
        )
        noise: Tensor = self.output_projection(output)
        return noise.squeeze(1)

    def __call__(self, noisy: Tensor, steps: Tensor, conditioning: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(noisy, steps, conditioning))
