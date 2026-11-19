# s_mamba

## Overview

S-Mamba (Wang et al., "Is Mamba Effective for Time Series Forecasting?", Neurocomputing 2025): the whole series of
each variate is embedded as one token, as in iTransformer, and two Mamba blocks read the variate tokens in opposite
orders in place of attention, followed by a feed-forward network.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `SMambaParameters`: sizes, layers, dropout and activation. |
| [encoder.py](./encoder.py) | `SMambaEncoder`: variate embedding and encoder layers. |
| [encoder_layer.py](./encoder_layer.py) | `SMambaEncoderLayer`: bidirectional Mamba and feed-forward network. |
| [forecast_network.py](./forecast_network.py) | `SMambaForecastNetwork`: each target variate's token to its horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `SMambaReconstructionNetwork`: each variate's token to its sequence. |
| [summary_network.py](./summary_network.py) | `SMambaSummaryNetwork`: all flattened tokens to the output vector. |
