# etsformer

## Overview

ETSformer (Woo et al., "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting", 2022): each
encoder layer extracts a seasonality with frequency attention (the top-k Fourier components, extrapolated) and a
growth with exponential smoothing attention, while the level of the input channels is exponentially smoothed; the
decoder damps the last growth of each layer over the horizon, and the forecast is level + growth + seasonality.
Forecasting and summaries (classification / regression, as in Time-Series-Library).

Differences from the paper and Time-Series-Library:

- The growth and seasonality are mapped to the target channels and every distribution parameter; the last level of
  each target channel is added only to the parameters in the scale of the values (`ForecastShape.level_parameters`).
- The embedding is the official causal convolution without position encoding, not Time-Series-Library's
  `DataEmbedding`; the feed-forward activation is the paper's sigmoid.
- Exponential smoothing is evaluated as a product with the lower-triangular weight matrix instead of an FFT
  convolution (same result). The auxiliary growth term of the level uses the convolution along time of the official
  code.
- The last encoder layer has no feed-forward network or normalizations, whose output no part of the model reads.
- The training-time jitter, scale and shift augmentation is left out; augmentation belongs to the task.
- No reconstruction network: Time-Series-Library's imputation extrapolates the window beyond itself instead of
  reconstructing its steps, and the in-sample level reads each observed value.
- Defaults follow the official multivariate ETTm2 script (salesforce/ETSformer `scripts/ETTm2.sh`: K = 3, with
  `run.py`'s model size 512, 8 heads, 2 encoder and decoder layers, feed-forward size 2048 and dropout 0.2); the
  official code has no ETTh1 script, and Time-Series-Library's ETTh1 script uses top_k 5 and dropout 0.1.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ETSformerParameters`: sizes, layers, top-k and dropout. |
| [exponential_smoothing.py](./exponential_smoothing.py) | `ETSformerExponentialSmoothing`: exponential smoothing attention with learnable rates. |
| [growth_layer.py](./growth_layer.py) | `ETSformerGrowthLayer`: smoothed successive differences of the tokens. |
| [frequency_attention.py](./frequency_attention.py) | `ETSformerFrequencyAttention`: top-k Fourier components, extrapolated. |
| [level_layer.py](./level_layer.py) | `ETSformerLevelLayer`: smoothed level of the input channels. |
| [damping_layer.py](./damping_layer.py) | `ETSformerDampingLayer`: last growth damped over the horizon. |
| [layer_output.py](./layer_output.py) | `ETSformerLayerOutput`: residual, level, growth and seasonality of a layer. |
| [encoder_layer.py](./encoder_layer.py) | `ETSformerEncoderLayer`: seasonality, growth, level and feed-forward network. |
| [encoding.py](./encoding.py) | `ETSformerEncoding`: final level and the growths and seasons of every layer. |
| [encoder.py](./encoder.py) | `ETSformerEncoder`: causal convolutional embedding and encoder layers. |
| [forecast_network.py](./forecast_network.py) | `ETSformerForecastNetwork`: level, damped growth and seasonality over the horizon. |
| [summary_network.py](./summary_network.py) | `ETSformerSummaryNetwork`: in-window growth and seasonality to the output vector. |
