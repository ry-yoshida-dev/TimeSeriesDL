from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .interpretation import TemporalFusionInterpretation
from .scalar_embedding import ScalarEmbedding
from .temporal_layers import TemporalFusionLayers
from .variable_selection_network import VariableSelectionNetwork

if TYPE_CHECKING:
    from .parameters import TemporalFusionTransformerParameters


class TemporalFusionTransformerForecastNetwork(ForecastNetwork):
    """
    Temporal Fusion Transformer over the past window and the horizon positions, without known or static covariates.

    Variable selection weighs the input channels of each past step; an LSTM encodes the past and a second LSTM,
    started from its final state, decodes learned horizon embeddings. Each horizon step attends causally to every
    past step and to the horizon steps up to itself, and a position-wise gated residual network maps it to the
    distribution parameters of every target channel.

    Parameters
    ----------
    parameters : TemporalFusionTransformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: TemporalFusionTransformerParameters, shape: ForecastShape) -> None:
        super().__init__()
        size: int = parameters.hidden_size
        self.embedding: ScalarEmbedding = ScalarEmbedding(shape.input_channels, size)
        self.selection: VariableSelectionNetwork = VariableSelectionNetwork(
            shape.input_channels, size, parameters.dropout
        )
        self.horizon_embedding: nn.Parameter = nn.Parameter(torch.randn(shape.horizon, size))
        self.layers: TemporalFusionLayers = TemporalFusionLayers(parameters, shape, is_enriched=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        outputs: Tensor = self._fuse(past)[0]
        return outputs

    def interpret(self, past: Tensor) -> TemporalFusionInterpretation:
        """
        Compute the variable selection and attention weights behind the forecasts of a batch of windows.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, input_length, input_channels), normalized as for ``forward``.

        Returns
        -------
        TemporalFusionInterpretation
            The weights.
        """
        interpretation: TemporalFusionInterpretation = self._fuse(past)[1]
        return interpretation

    def _fuse(self, past: Tensor) -> tuple[Tensor, TemporalFusionInterpretation]:
        selected: Tensor
        variable_weights: Tensor
        selected, variable_weights = self.selection(self.embedding(past))
        future: Tensor = self.horizon_embedding.unsqueeze(0).expand(past.shape[0], -1, -1)
        outputs: Tensor
        attention_weights: Tensor
        outputs, attention_weights = self.layers(selected, future)
        return outputs, TemporalFusionInterpretation(variable_weights, attention_weights)
