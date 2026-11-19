# non_stationary_transformer

## Overview

Non-stationary Transformer (Liu et al., "Non-stationary Transformers: Exploring the Stationarity in Time Series
Forecasting", NeurIPS 2022): each window is stationarized by its own mean and standard deviation, and de-stationary
attention rescales the scores by a factor tau and shifts them by a factor delta per input step, both learned from the
raw series and its statistics, to recover the attention of the non-stationary series. The backbone is the vanilla
Transformer encoder-decoder.

Differences from the paper and Time-Series-Library:

- The model reads the windows it is given. When the forecasting task also normalizes each window (RevIN), the
  statistics reaching the projectors are close to zero and one; disable the task's instance normalization to let
  tau and delta learn from the raw statistics as in the paper.
- Only the level parameters of each target channel (such as the mean or the quantiles) are restored to the scale of
  the window; the other distribution parameters stay in the stationarized scale.
- Reconstructions stationarize the whole sequence, as Time-Series-Library's anomaly detection does; the masked
  statistics of its imputation are not used, since missing values are not marked.
- Summaries embed the raw sequence, as Time-Series-Library's classification does.
- The token embedding has no calendar (time stamp) embedding, as elsewhere in this package.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `NonStationaryTransformerParameters`: sizes, layers, projector sizes and label length. |
| [stationarization.py](./stationarization.py) | `Stationarization`: per-window statistics, normalization and restoration. |
| [de_stationary_factors.py](./de_stationary_factors.py) | `DeStationaryFactors`: tau and delta of a batch of windows. |
| [factor_projector.py](./factor_projector.py) | `NonStationaryFactorProjector`: MLP from the raw series and one statistic to a factor. |
| [factor_learner.py](./factor_learner.py) | `DeStationaryFactorLearner`: tau from the standard deviation, delta from the mean. |
| [de_stationary_attention.py](./de_stationary_attention.py) | `DeStationaryAttention`: multi-head attention with scores rescaled by tau and shifted by delta. |
| [encoder_layer.py](./encoder_layer.py) | `NonStationaryEncoderLayer`: de-stationary self-attention and feed-forward network. |
| [decoder_layer.py](./decoder_layer.py) | `NonStationaryDecoderLayer`: causal self-attention, cross-attention and feed-forward network. |
| [encoder.py](./encoder.py) | `NonStationaryEncoder`: embedding, encoder layers and final normalization. |
| [forecast_network.py](./forecast_network.py) | `NonStationaryForecastNetwork`: encoder-decoder forecast of the whole horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `NonStationaryReconstructionNetwork`: each step's tokens to its values. |
| [summary_network.py](./summary_network.py) | `NonStationarySummaryNetwork`: all flattened tokens to the output vector. |
