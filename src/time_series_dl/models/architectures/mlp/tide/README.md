# tide

## Overview

TiDE (Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR 2023): a channel-independent
dense encoder and decoder of each target series, with a linear residual from the window to the horizon. With
covariates it follows the paper: a feature projection maps each step's known covariates to a few features, which the
dense encoder reads for the whole window and horizon together with the static attributes (categorical ones embedded),
and the temporal decoder reads for its own step. Past-observed channels that are not targets are not read.
Forecasting only.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TiDEParameters`: hidden sizes, layer counts, dropout, layer normalization and covariate projection size. |
| [residual_block.py](./residual_block.py) | `TiDEResidualBlock`: MLP with a linear skip and layer normalization. |
| [dense_encoder_decoder.py](./dense_encoder_decoder.py) | `TiDEDenseEncoderDecoder`: dense encoder, dense decoder, temporal decoder and linear residual. |
| [forecast_network.py](./forecast_network.py) | `TiDEForecastNetwork`: encodes each target window and decodes it to the horizon. |
| [covariate_forecast_network.py](./covariate_forecast_network.py) | `TiDECovariateForecastNetwork`: adds the feature projection of known covariates and the static attributes. |
