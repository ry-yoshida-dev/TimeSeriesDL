from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import ETSformerEncoder

if TYPE_CHECKING:
    from .parameters import ETSformerParameters


class ETSformerSummaryNetwork(SummaryNetwork):
    """
    ETSformer encoder without the level, whose growth and seasonality over the input window, summed over the layers,
    pass through GELU and dropout and are mapped linearly from all flattened steps to the output vector, as in
    Time-Series-Library's classification.

    Parameters
    ----------
    parameters : ETSformerParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: ETSformerParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.input_length: int = shape.input_length
        self.encoder: ETSformerEncoder = ETSformerEncoder(parameters, shape.input_channels, 0, is_level_tracked=False)
        self.activation: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.model_size * shape.input_length, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        encoding = self.encoder(values)
        components: list[Tensor] = [growth[:, : self.input_length] for growth in encoding.growths]
        components.extend(season[:, : self.input_length] for season in encoding.seasons)
        features: Tensor = torch.stack(components).sum(0)
        outputs: Tensor = self.projection(self.dropout(self.activation(features)).flatten(start_dim=1))
        return outputs
