from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.attention.value_embedding import ValueEmbedding
from .coarser_scale_construction import PyraformerCoarserScaleConstruction
from .encoder_layer import PyraformerEncoderLayer
from .pyramid import PyraformerPyramid

if TYPE_CHECKING:
    from .parameters import PyraformerParameters


class PyraformerEncoder(nn.Module):
    """
    Pyraformer encoder: value embedding, coarser-scale construction, pyramidal attention layers, and for every input
    step the concatenated features of its ancestors at all scales.

    Inputs shorter than the product of the window sizes, for which a scale would be empty, are padded at the start
    with copies of their first step.

    Parameters
    ----------
    parameters : PyraformerParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If input_length is not positive.
    """

    def __init__(self, parameters: PyraformerParameters, input_length: int, input_channels: int) -> None:
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        super().__init__()
        self.padded_length: int = max(input_length, math.prod(parameters.window_sizes))
        self.padding: int = self.padded_length - input_length
        pyramid: PyraformerPyramid = PyraformerPyramid(
            input_length=self.padded_length, window_sizes=parameters.window_sizes, inner_size=parameters.inner_size
        )
        self.feature_size: int = (len(parameters.window_sizes) + 1) * parameters.model_size
        self.embedding: ValueEmbedding = ValueEmbedding(
            input_channels, parameters.model_size, parameters.dropout, self.padded_length
        )
        self.construction: PyraformerCoarserScaleConstruction = PyraformerCoarserScaleConstruction(
            parameters.model_size, parameters.bottleneck_size, parameters.window_sizes
        )
        mask: Tensor = pyramid.build_mask()
        self.layers: nn.Sequential = nn.Sequential(
            *(
                PyraformerEncoderLayer(
                    mask, parameters.model_size, parameters.num_heads, parameters.feedforward_size, parameters.dropout
                )
                for _ in range(parameters.num_layers)
            )
        )
        self.reference_indexes: Tensor
        self.register_buffer("reference_indexes", pyramid.build_reference_indexes().flatten(), persistent=False)

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
            Features with shape (batch, padded_length, feature_size).
        """
        padded: Tensor = torch.cat([values[:, :1].expand(-1, self.padding, -1), values], dim=1)
        nodes: Tensor = self.layers(self.construction(self.embedding(padded)))
        gathered: Tensor = nodes.index_select(1, self.reference_indexes)
        return gathered.reshape(values.shape[0], self.padded_length, self.feature_size)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
