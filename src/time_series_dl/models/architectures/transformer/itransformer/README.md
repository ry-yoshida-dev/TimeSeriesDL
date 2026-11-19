# itransformer

## Overview

iTransformer (Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
ICLR 2024): the whole series of each variate is embedded as one token, and a Transformer encoder attends across
variates.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ITransformerParameters`: Transformer sizes, dropout and activation. |
| [encoder.py](./encoder.py) | `ITransformerEncoder`: variate embedding and encoder layers. |
| [forecast_network.py](./forecast_network.py) | `ITransformerForecastNetwork`: each target variate's token to its horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `ITransformerReconstructionNetwork`: each variate's token to its sequence. |
| [summary_network.py](./summary_network.py) | `ITransformerSummaryNetwork`: all flattened tokens to the output vector. |
