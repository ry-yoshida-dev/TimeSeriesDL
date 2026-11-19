from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from ....components.attention.value_embedding import ValueEmbedding
from .isometric_layer import MICNIsometricLayer

if TYPE_CHECKING:
    from .parameters import MICNParameters


class MICNSeasonalPrediction(nn.Module):
    """
    Seasonal prediction block of MICN: token embedding with position encoding, MIC layers and a linear projection of
    each step.

    Parameters
    ----------
    parameters : MICNParameters
        The model configuration.
    input_channels : int
        Number of channels of the seasonal input.
    output_size : int
        Number of outputs per time step.
    length : int
        Number of time steps of the sequence.
    """

    def __init__(self, parameters: MICNParameters, input_channels: int, output_size: int, length: int) -> None:
        super().__init__()
        self.embedding: ValueEmbedding = ValueEmbedding(
            input_channels, parameters.model_size, parameters.dropout, length
        )
        self.layers: nn.Sequential = nn.Sequential(
            *(MICNIsometricLayer(parameters, length) for _ in range(parameters.num_layers))
        )
        self.projection: nn.Linear = nn.Linear(parameters.model_size, output_size)

    def forward(self, seasonal: Tensor) -> Tensor:
        """
        Predict the seasonal part of every step.

        Parameters
        ----------
        seasonal : Tensor
            Seasonal values with shape (batch, length, input_channels).

        Returns
        -------
        Tensor
            Outputs with shape (batch, length, output_size).
        """
        outputs: Tensor = self.projection(self.layers(self.embedding(seasonal)))
        return outputs

    def __call__(self, seasonal: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(seasonal))
