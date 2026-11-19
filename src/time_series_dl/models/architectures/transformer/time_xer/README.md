# time_xer

## Overview

TimeXer (Wang et al., "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables",
NeurIPS 2024): each endogenous series is embedded as non-overlapping patch tokens plus a learnable global token, each
exogenous series as one variate token from its whole series; self-attention mixes the patch and global tokens of
each endogenous variate, and cross-attention brings the exogenous information into its global token. Forecasting
only.

The target channels are the endogenous variates and the other input channels the exogenous ones. When every input
channel is a target, every channel is both endogenous and exogenous, as in the paper's multivariate setting.

Differences from the paper and Time-Series-Library:

- An input length that is not a multiple of the patch length is padded at the start with copies of the first value,
  instead of dropping the last steps.
- No calendar covariates are available, so there are no time-stamp tokens among the exogenous tokens.
- The head maps to every distribution parameter of the horizon; instance normalization is left to the forecaster.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimeXerParameters`: patch length, sizes, layers, dropout and activation. |
| [endogenous_embedding.py](./endogenous_embedding.py) | `TimeXerEndogenousEmbedding`: patch tokens and a global token per endogenous variate. |
| [encoder_layer.py](./encoder_layer.py) | `TimeXerEncoderLayer`: self-attention, global-to-exogenous cross-attention and feed-forward network. |
| [forecast_network.py](./forecast_network.py) | `TimeXerForecastNetwork`: endogenous and exogenous embeddings, encoder and flatten head. |
