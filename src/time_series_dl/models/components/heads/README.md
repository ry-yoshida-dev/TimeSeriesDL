# heads

## Overview

Generic networks of every task around any `SequenceEncoder`, so an encoder-based model only defines its encoder.

## Components

| Component | Description |
| --------- | ----------- |
| [head.py](./head.py) | `EncoderHead`: pooling and dropout of the heads, and the builders of the three networks. |
| [forecast_network.py](./forecast_network.py) | `EncodedForecastNetwork`: flattened features mapped linearly to the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `EncodedReconstructionNetwork`: each step's features mapped linearly to its values. |
| [summary_network.py](./summary_network.py) | `EncodedSummaryNetwork`: pooled features mapped linearly to the output vector. |
