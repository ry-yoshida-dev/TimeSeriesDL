# seg_rnn

## Overview

SegRNN (Lin et al., "SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting", 2023): a
channel-independent GRU over segments of each target series, decoded in parallel for every output segment.
Forecasting only.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `SegRNNParameters`: segment length, hidden size, dropout and channel embeddings. |
| [forecast_network.py](./forecast_network.py) | `SegRNNForecastNetwork`: encodes the segments and decodes every output segment at once. |
