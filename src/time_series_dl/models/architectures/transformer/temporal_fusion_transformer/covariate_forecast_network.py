from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import CovariateForecastNetwork, CovariateForecastShape, CovariateTensors
from .interpretation import TemporalFusionInterpretation
from .scalar_embedding import ScalarEmbedding
from .static_contexts import StaticContexts
from .static_covariate_encoder import StaticCovariateEncoder
from .temporal_layers import TemporalFusionLayers
from .variable_selection_network import VariableSelectionNetwork

if TYPE_CHECKING:
    from .parameters import TemporalFusionTransformerParameters


class TemporalFusionTransformerCovariateForecastNetwork(CovariateForecastNetwork):
    """
    Temporal Fusion Transformer with known and static covariates, as in the paper.

    Static features pass through the static covariate encoders, whose contexts condition both variable selection
    networks, start the LSTM encoder and enrich every step before attention. The past variable selection weighs the
    input channels and the known covariates of each past step; the future variable selection weighs the known
    covariates of each horizon step, which the LSTM decoder reads. Without known covariates the decoder reads learned
    horizon embeddings, and without static features the contexts and static enrichment are omitted.

    Parameters
    ----------
    parameters : TemporalFusionTransformerParameters
        The model configuration.
    shape : CovariateForecastShape
        The dimensions of the forecast and of the covariates.
    """

    def __init__(self, parameters: TemporalFusionTransformerParameters, shape: CovariateForecastShape) -> None:
        super().__init__()
        size: int = parameters.hidden_size
        dropout: float = parameters.dropout
        forecast = shape.forecast
        context_size: int = size if shape.is_static_present else 0
        self.input_length: int = forecast.input_length
        self.num_lstm_layers: int = parameters.num_lstm_layers
        past_inputs: int = forecast.input_channels + shape.known_channels
        self.static_encoder: StaticCovariateEncoder | None = (
            StaticCovariateEncoder(shape.static_channels, shape.static_cardinalities, size, dropout)
            if shape.is_static_present
            else None
        )
        self.past_embedding: ScalarEmbedding = ScalarEmbedding(past_inputs, size)
        self.past_selection: VariableSelectionNetwork = VariableSelectionNetwork(
            past_inputs, size, dropout, context_size
        )
        self.future_embedding: ScalarEmbedding | None = None
        self.future_selection: VariableSelectionNetwork | None = None
        self.horizon_embedding: nn.Parameter | None = None
        if shape.is_known_present:
            self.future_embedding = ScalarEmbedding(shape.known_channels, size)
            self.future_selection = VariableSelectionNetwork(shape.known_channels, size, dropout, context_size)
        else:
            self.horizon_embedding = nn.Parameter(torch.randn(forecast.horizon, size))
        self.layers: TemporalFusionLayers = TemporalFusionLayers(
            parameters, forecast, is_enriched=shape.is_static_present
        )

    @override
    def forward(self, past: Tensor, covariates: CovariateTensors, future_targets: Tensor | None = None) -> Tensor:
        outputs: Tensor = self._fuse(past, covariates)[0]
        return outputs

    def interpret(self, past: Tensor, covariates: CovariateTensors) -> TemporalFusionInterpretation:
        """
        Compute the variable selection and attention weights behind the forecasts of a batch of windows.

        Parameters
        ----------
        past : Tensor
            Past values with shape (batch, input_length, input_channels), normalized as for ``forward``.
        covariates : CovariateTensors
            The covariates, as for ``forward``.

        Returns
        -------
        TemporalFusionInterpretation
            The weights.
        """
        interpretation: TemporalFusionInterpretation = self._fuse(past, covariates)[1]
        return interpretation

    def _fuse(self, past: Tensor, covariates: CovariateTensors) -> tuple[Tensor, TemporalFusionInterpretation]:
        contexts: StaticContexts | None = (
            None
            if self.static_encoder is None
            else self.static_encoder(covariates.static, covariates.static_categories)
        )
        selection_context: Tensor | None = None if contexts is None else contexts.selection
        known_past: Tensor = covariates.known[:, : self.input_length]
        selected_past: Tensor
        past_weights: Tensor
        selected_past, past_weights = self.past_selection(
            self.past_embedding(torch.cat([past, known_past], dim=2)), selection_context
        )
        future: Tensor
        future_weights: Tensor | None = None
        if self.future_embedding is not None and self.future_selection is not None:
            future, future_weights = self.future_selection(
                self.future_embedding(covariates.known[:, self.input_length :]), selection_context
            )
        elif self.horizon_embedding is not None:
            future = self.horizon_embedding.unsqueeze(0).expand(past.shape[0], -1, -1)
        else:
            raise AssertionError("the network has either known covariates or horizon embeddings.")
        initial_state: tuple[Tensor, Tensor] | None = (
            None if contexts is None else (self._repeat_layers(contexts.hidden), self._repeat_layers(contexts.cell))
        )
        outputs: Tensor
        attention_weights: Tensor
        outputs, attention_weights = self.layers(
            selected_past, future, initial_state, None if contexts is None else contexts.enrichment
        )
        return outputs, TemporalFusionInterpretation(
            variable_weights=past_weights,
            attention_weights=attention_weights,
            future_variable_weights=future_weights,
            static_variable_weights=None if contexts is None else contexts.variable_weights,
        )

    def _repeat_layers(self, state: Tensor) -> Tensor:
        return state.unsqueeze(0).expand(self.num_lstm_layers, -1, -1).contiguous()
