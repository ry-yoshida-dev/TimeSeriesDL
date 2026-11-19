from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....interfaces import ForecastShape
from .gate_add_norm import GateAddNorm
from .gated_residual_network import GatedResidualNetwork
from .interpretable_multi_head_attention import InterpretableMultiHeadAttention

if TYPE_CHECKING:
    from .parameters import TemporalFusionTransformerParameters


class TemporalFusionLayers(nn.Module):
    """
    Layers of TFT after variable selection: the LSTM sequence-to-sequence layer with a gated skip connection, static
    enrichment, interpretable self-attention of each horizon step, the position-wise feed-forward network, and the
    output layer.

    Parameters
    ----------
    parameters : TemporalFusionTransformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    is_enriched : bool
        Whether a static enrichment context is added to every step before attention.
    """

    def __init__(
        self, parameters: TemporalFusionTransformerParameters, shape: ForecastShape, is_enriched: bool
    ) -> None:
        super().__init__()
        size: int = parameters.hidden_size
        dropout: float = parameters.dropout
        lstm_dropout: float = dropout if parameters.num_lstm_layers > 1 else 0.0
        self.input_length: int = shape.input_length
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.encoder: nn.LSTM = nn.LSTM(size, size, parameters.num_lstm_layers, batch_first=True, dropout=lstm_dropout)
        self.decoder: nn.LSTM = nn.LSTM(size, size, parameters.num_lstm_layers, batch_first=True, dropout=lstm_dropout)
        self.sequence_gate: GateAddNorm = GateAddNorm(size, size, dropout)
        self.attention: InterpretableMultiHeadAttention = InterpretableMultiHeadAttention(
            size, parameters.num_heads, dropout
        )
        self.attention_gate: GateAddNorm = GateAddNorm(size, size, dropout)
        self.feed_forward: GatedResidualNetwork = GatedResidualNetwork(size, size, size, dropout)
        self.output_gate: GateAddNorm = GateAddNorm(size, size, 0.0)
        self.head: nn.Linear = nn.Linear(size, shape.output_channels * shape.num_parameters)
        self.enrichment: GatedResidualNetwork | None = (
            GatedResidualNetwork(size, size, size, dropout, context_size=size) if is_enriched else None
        )
        horizon_positions: Tensor = torch.arange(shape.horizon).unsqueeze(1) + shape.input_length
        key_positions: Tensor = torch.arange(shape.input_length + shape.horizon).unsqueeze(0)
        self.is_masked: Tensor
        self.register_buffer("is_masked", key_positions > horizon_positions, persistent=False)

    def forward(
        self,
        past: Tensor,
        future: Tensor,
        initial_state: tuple[Tensor, Tensor] | None = None,
        enrichment_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forecast from the selected inputs of the past window and of the horizon.

        Parameters
        ----------
        past : Tensor
            Selected past inputs with shape (batch, input_length, hidden_size).
        future : Tensor
            Selected horizon inputs with shape (batch, horizon, hidden_size).
        initial_state : tuple[Tensor, Tensor] | None
            Hidden and cell states that start the LSTM encoder, each with shape (num_lstm_layers, batch, hidden_size);
            zeros when None.
        enrichment_context : Tensor | None
            Static enrichment context with shape (batch, hidden_size); required exactly when ``is_enriched``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Unconstrained distribution parameters with shape (batch, horizon, output_channels, num_parameters), and the
            attention weights with shape (batch, horizon, input_length + horizon).
        """
        batch_size: int = past.shape[0]
        encoded: Tensor
        state: tuple[Tensor, Tensor]
        encoded, state = self.encoder(past, initial_state)
        decoded: Tensor = self.decoder(future, state)[0]
        temporal: Tensor = self.sequence_gate(torch.cat([encoded, decoded], dim=1), torch.cat([past, future], dim=1))
        enriched: Tensor = temporal if self.enrichment is None else self.enrichment(temporal, enrichment_context)
        enriched_future: Tensor = enriched[:, self.input_length :]
        attended: Tensor
        attention_weights: Tensor
        attended, attention_weights = self.attention(enriched_future, enriched, self.is_masked)
        attention_output: Tensor = self.attention_gate(attended, enriched_future)
        fused: Tensor = self.output_gate(self.feed_forward(attention_output), temporal[:, self.input_length :])
        outputs: Tensor = self.head(fused).reshape(batch_size, self.horizon, self.output_channels, self.num_parameters)
        return outputs, attention_weights

    def __call__(
        self,
        past: Tensor,
        future: Tensor,
        initial_state: tuple[Tensor, Tensor] | None = None,
        enrichment_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(past, future, initial_state, enrichment_context))
