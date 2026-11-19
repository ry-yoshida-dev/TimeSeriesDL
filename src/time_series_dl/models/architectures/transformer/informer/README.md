# informer

## Overview

Informer (Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021):
ProbSparse self-attention, self-attention distilling that halves the length between encoder layers, and a
generative-style decoder that outputs the whole horizon in one pass. Reconstructions and summaries use the encoder
without distilling and the generic heads. The stack of encoders on halved inputs of the paper is left out.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `InformerParameters`: sizes, layers, sampling factor, distilling, label length and head. |
| [encoder_layer.py](./encoder_layer.py) | `InformerEncoderLayer`: self-attention and feed-forward network. |
| [distilling_layer.py](./distilling_layer.py) | `InformerDistillingLayer`: convolution and max-pooling that halve the length. |
| [decoder_layer.py](./decoder_layer.py) | `InformerDecoderLayer`: masked self-attention, cross-attention and feed-forward network. |
| [sequence_encoder.py](./sequence_encoder.py) | `InformerSequenceEncoder`: embedding, encoder layers and optional distilling. |
| [forecast_network.py](./forecast_network.py) | `InformerForecastNetwork`: encoder-decoder forecast of the whole horizon. |
