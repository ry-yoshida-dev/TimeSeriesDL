# temporal_fusion_transformer

## Overview

Temporal Fusion Transformer (Lim et al., "Temporal Fusion Transformers for interpretable multi-horizon time series
forecasting", International Journal of Forecasting, 2021): variable selection over the inputs, an LSTM encoder and
decoder, and interpretable multi-head attention, joined by gated skip connections. With covariates it follows the
paper: static covariate encoders give the contexts of variable selection, of static enrichment and of the initial
LSTM state, past known covariates join the past inputs, and future known covariates are the decoder's inputs.
Without static features the contexts and static enrichment are omitted, and without known covariates the decoder
reads learned horizon embeddings. Forecasts are not sampled; quantiles come from the task's objective.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TemporalFusionTransformerParameters`: hidden size, heads, LSTM layers and dropout. |
| [forecast_network.py](./forecast_network.py) | `TemporalFusionTransformerForecastNetwork`: forecasts without known or static covariates, and explains them. |
| [covariate_forecast_network.py](./covariate_forecast_network.py) | `TemporalFusionTransformerCovariateForecastNetwork`: forecasts with known and static covariates, and explains them. |
| [temporal_layers.py](./temporal_layers.py) | `TemporalFusionLayers`: LSTM sequence-to-sequence layer, static enrichment, attention and output layers. |
| [static_covariate_encoder.py](./static_covariate_encoder.py) | `StaticCovariateEncoder`: selects the static features and derives the four contexts. |
| [static_contexts.py](./static_contexts.py) | `StaticContexts`: the contexts and static selection weights. |
| [interpretation.py](./interpretation.py) | `TemporalFusionInterpretation`: variable selection and attention weights. |
| [variable_selection_network.py](./variable_selection_network.py) | `VariableSelectionNetwork`: weighs embedded variables, optionally given a static context. |
| [scalar_embedding.py](./scalar_embedding.py) | `ScalarEmbedding`: linear embedding of each real-valued channel. |
| [interpretable_multi_head_attention.py](./interpretable_multi_head_attention.py) | `InterpretableMultiHeadAttention`: heads sharing their values. |
| [gated_residual_network.py](./gated_residual_network.py) | `GatedResidualNetwork`: ELU layers with an optional context and a gated skip connection. |
| [gate_add_norm.py](./gate_add_norm.py) | `GateAddNorm`: gated linear unit, residual addition and layer normalization. |

## Examples

```python
from time_series_dl.models import CovariateForecastShape, ForecastShape
from time_series_dl.models.temporal_fusion_transformer import TemporalFusionTransformerParameters

shape = ForecastShape(input_length=48, input_channels=3, horizon=12, target_channels=(0,))
network = TemporalFusionTransformerParameters(hidden_size=32).build_forecast_network(shape)
covariate_shape = CovariateForecastShape(forecast=shape, known_channels=4, static_cardinalities=(10,))
covariate_network = TemporalFusionTransformerParameters(hidden_size=32).build_covariate_forecast_network(
    covariate_shape
)
```
