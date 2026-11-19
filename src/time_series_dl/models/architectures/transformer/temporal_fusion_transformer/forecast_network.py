from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .gate_add_norm import GateAddNorm
from .gated_residual_network import GatedResidualNetwork
from .interpretable_multi_head_attention import InterpretableMultiHeadAttention
from .interpretation import TemporalFusionInterpretation
from .variable_selection_network import VariableSelectionNetwork

if TYPE_CHECKING:
    from .parameters import TemporalFusionTransformerParameters


class TemporalFusionTransformerForecastNetwork(ForecastNetwork):
    """
    Temporal Fusion Transformer over the past window and the horizon positions.

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
        dropout: float = parameters.dropout
        lstm_dropout: float = dropout if parameters.num_lstm_layers > 1 else 0.0
        self.input_length: int = shape.input_length
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.selection: VariableSelectionNetwork = VariableSelectionNetwork(shape.input_channels, size, dropout)
        self.horizon_embedding: nn.Parameter = nn.Parameter(torch.randn(shape.horizon, size))
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
        horizon_positions: Tensor = torch.arange(shape.horizon).unsqueeze(1) + shape.input_length
        key_positions: Tensor = torch.arange(shape.input_length + shape.horizon).unsqueeze(0)
        self.is_masked: Tensor
        self.register_buffer("is_masked", key_positions > horizon_positions, persistent=False)

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
        batch_size: int = past.shape[0]
        selected: Tensor
        variable_weights: Tensor
        selected, variable_weights = self.selection(past)
        future: Tensor = self.horizon_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        encoded: Tensor
        state: tuple[Tensor, Tensor]
        encoded, state = self.encoder(selected)
        decoded: Tensor = self.decoder(future, state)[0]
        temporal: Tensor = self.sequence_gate(
            torch.cat([encoded, decoded], dim=1), torch.cat([selected, future], dim=1)
        )
        temporal_future: Tensor = temporal[:, self.input_length :]
        attended: Tensor
        attention_weights: Tensor
        attended, attention_weights = self.attention(temporal_future, temporal, self.is_masked)
        enriched: Tensor = self.attention_gate(attended, temporal_future)
        fused: Tensor = self.output_gate(self.feed_forward(enriched), temporal_future)
        outputs: Tensor = self.head(fused).reshape(batch_size, self.horizon, self.output_channels, self.num_parameters)
        return outputs, TemporalFusionInterpretation(variable_weights, attention_weights)
