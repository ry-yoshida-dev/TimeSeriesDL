# temporal_fusion_transformer

## Overview

Temporal Fusion Transformer (Lim et al., "Temporal Fusion Transformers for interpretable multi-horizon time series
forecasting", International Journal of Forecasting, 2021): variable selection over the input channels, an LSTM
encoder and decoder, and interpretable multi-head attention, joined by gated skip connections. Without static
covariates or known future inputs, static enrichment is omitted and the decoder reads learned horizon embeddings.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TemporalFusionTransformerParameters`: hidden size, heads, LSTM layers and dropout. |
| [forecast_network.py](./forecast_network.py) | `TemporalFusionTransformerForecastNetwork`: forecasts and explains them. |
| [interpretation.py](./interpretation.py) | `TemporalFusionInterpretation`: variable selection and attention weights. |
| [variable_selection_network.py](./variable_selection_network.py) | `VariableSelectionNetwork`: weighs the channels of each step. |
| [interpretable_multi_head_attention.py](./interpretable_multi_head_attention.py) | `InterpretableMultiHeadAttention`: heads sharing their values. |
| [gated_residual_network.py](./gated_residual_network.py) | `GatedResidualNetwork`: ELU layers with a gated skip connection. |
| [gate_add_norm.py](./gate_add_norm.py) | `GateAddNorm`: gated linear unit, residual addition and layer normalization. |

## Examples

```python
from time_series_dl.models import ForecastShape
from time_series_dl.models.temporal_fusion_transformer import TemporalFusionTransformerParameters

shape = ForecastShape(input_length=48, input_channels=3, horizon=12, target_channels=(0,))
network = TemporalFusionTransformerParameters(hidden_size=32).build_forecast_network(shape)
```
