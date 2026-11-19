# tide

## Overview

TiDE (Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR 2023) without covariates: a
channel-independent dense encoder and decoder of each target series, with a linear residual from the window to the
horizon. Forecasting only.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TiDEParameters`: hidden sizes, layer counts, dropout and layer normalization. |
| [residual_block.py](./residual_block.py) | `TiDEResidualBlock`: MLP with a linear skip and layer normalization. |
| [forecast_network.py](./forecast_network.py) | `TiDEForecastNetwork`: encodes each target window and decodes it to the horizon. |
