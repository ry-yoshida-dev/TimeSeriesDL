from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .parameters import CSDIParameters


class CSDIResidualLayer(nn.Module):
    """
    Residual layer of the CSDI denoiser: the diffusion step embedding is added, a Transformer layer attends along time
    for each feature and another along the features at each time, and the side information enters a tanh-sigmoid
    gate that yields the residual and skip outputs.

    Parameters
    ----------
    parameters : CSDIParameters
        The model configuration.
    side_size : int
        Number of side information channels.
    is_feature_attended : bool
        Whether the layer attends along the features; False for a single feature, where that attention is the
        identity.
    """

    def __init__(self, parameters: CSDIParameters, side_size: int, is_feature_attended: bool) -> None:
        super().__init__()
        channels: int = parameters.channels
        self.step_projection: nn.Linear = nn.Linear(parameters.step_feature_size, channels)
        self.time_layer: nn.TransformerEncoderLayer = self._build_transformer(parameters)
        self.feature_layer: nn.TransformerEncoderLayer | None = (
            self._build_transformer(parameters) if is_feature_attended else None
        )
        self.middle_projection: nn.Conv1d = nn.Conv1d(channels, 2 * channels, 1)
        self.side_projection: nn.Conv1d = nn.Conv1d(side_size, 2 * channels, 1)
        self.output_projection: nn.Conv1d = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, values: Tensor, side: Tensor, step_embedding: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply the layer.

        Parameters
        ----------
        values : Tensor
            Hidden values with shape (batch, channels, features, length).
        side : Tensor
            Side information with shape (batch, side_size, features, length).
        step_embedding : Tensor
            Diffusion step embedding with shape (batch, step_feature_size).

        Returns
        -------
        tuple[Tensor, Tensor]
            The residual output and the skip output, each with the shape of ``values``.
        """
        batch: int
        channels: int
        features: int
        length: int
        batch, channels, features, length = values.shape
        hidden: Tensor = values + self.step_projection(step_embedding)[:, :, None, None]
        # Attention along time: one sequence of length ``length`` per (window, feature).
        hidden = (
            self.time_layer(hidden.permute(0, 2, 3, 1).reshape(batch * features, length, channels))
            .reshape(batch, features, length, channels)
            .permute(0, 3, 1, 2)
        )
        if self.feature_layer is not None:
            hidden = (
                self.feature_layer(hidden.permute(0, 3, 2, 1).reshape(batch * length, features, channels))
                .reshape(batch, length, features, channels)
                .permute(0, 3, 2, 1)
            )
        flat: Tensor = self.middle_projection(hidden.reshape(batch, channels, features * length))
        flat = flat + self.side_projection(side.reshape(batch, side.shape[1], features * length))
        gate: Tensor
        content: Tensor
        gate, content = flat.chunk(2, dim=1)
        flat = self.output_projection(torch.sigmoid(gate) * torch.tanh(content))
        residual: Tensor
        skip: Tensor
        residual, skip = flat.reshape(batch, 2 * channels, features, length).chunk(2, dim=1)
        return (values + residual) / math.sqrt(2.0), skip

    @staticmethod
    def _build_transformer(parameters: CSDIParameters) -> nn.TransformerEncoderLayer:
        return nn.TransformerEncoderLayer(
            parameters.channels,
            parameters.num_heads,
            parameters.feedforward_size,
            parameters.dropout,
            activation="gelu",
            batch_first=True,
        )

    def __call__(self, values: Tensor, side: Tensor, step_embedding: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values, side, step_embedding))
