from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.diffusion import DiffusionStepEmbedding
from .residual_layer import CSDIResidualLayer
from .side_information import CSDISideInformation

if TYPE_CHECKING:
    from .parameters import CSDIParameters


class CSDIDenoiser(nn.Module):
    """
    Noise predictor of CSDI: predicts the noise of the unconditioned entries of a (features, time) array from the
    conditioned (observed) entries, the noisy unconditioned entries, the conditional mask and the diffusion step.

    The conditioned values and the noisy values form two input channels; every entry also reads its side information
    (time and feature embeddings and the mask), so the denoiser works with any mask, as imputation needs.

    Parameters
    ----------
    parameters : CSDIParameters
        The model configuration.
    num_features : int
        Number of features of the arrays.
    """

    def __init__(self, parameters: CSDIParameters, num_features: int) -> None:
        super().__init__()
        self.side_information: CSDISideInformation = CSDISideInformation(parameters, num_features)
        self.step_embedding: DiffusionStepEmbedding = DiffusionStepEmbedding(
            parameters.step_feature_size, parameters.step_feature_size
        )
        self.input_projection: nn.Conv1d = nn.Conv1d(2, parameters.channels, 1)
        self.layers: nn.ModuleList = nn.ModuleList(
            CSDIResidualLayer(parameters, self.side_information.size, num_features > 1)
            for _ in range(parameters.num_layers)
        )
        self.skip_projection: nn.Conv1d = nn.Conv1d(parameters.channels, parameters.channels, 1)
        self.output_projection: nn.Conv1d = nn.Conv1d(parameters.channels, 1, 1)
        _ = nn.init.zeros_(self.output_projection.weight)

    def forward(self, conditional_values: Tensor, noisy: Tensor, conditional_mask: Tensor, steps: Tensor) -> Tensor:
        """
        Predict the noise.

        Parameters
        ----------
        conditional_values : Tensor
            Observed values at conditioned entries and zero elsewhere, with shape (batch, features, length).
        noisy : Tensor
            Noisy values at unconditioned entries and zero elsewhere, with the same shape.
        conditional_mask : Tensor
            One for conditioned entries and zero otherwise, with the same shape.
        steps : Tensor
            Zero-based diffusion steps with shape (batch,).

        Returns
        -------
        Tensor
            The predicted noise with shape (batch, features, length).
        """
        batch: int
        features: int
        length: int
        batch, features, length = noisy.shape
        inputs: Tensor = torch.stack([conditional_values, noisy], dim=1).reshape(batch, 2, features * length)
        hidden: Tensor = functional.relu(self.input_projection(inputs)).reshape(batch, -1, features, length)
        side: Tensor = self.side_information(conditional_mask)
        step_embedding: Tensor = self.step_embedding(steps)
        skips: Tensor = hidden.new_zeros(hidden.shape)
        for layer in self.layers:
            skip: Tensor
            hidden, skip = cast(CSDIResidualLayer, layer)(hidden, side, step_embedding)
            skips = skips + skip
        flat: Tensor = (skips / math.sqrt(len(self.layers))).reshape(batch, -1, features * length)
        output: Tensor = self.output_projection(functional.relu(self.skip_projection(flat)))
        return output.reshape(batch, features, length)

    def __call__(self, conditional_values: Tensor, noisy: Tensor, conditional_mask: Tensor, steps: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(conditional_values, noisy, conditional_mask, steps))
