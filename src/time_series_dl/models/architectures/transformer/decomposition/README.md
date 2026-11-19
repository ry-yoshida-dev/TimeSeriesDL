# decomposition

## Overview

The decomposition Transformer architecture shared by Autoformer (Wu et al., NeurIPS 2021) and FEDformer (Zhou et al.,
ICML 2022): encoder and decoder layers with series decomposition blocks, a seasonal decoder input and an accumulated
trend. The variants differ only in the attention mechanisms their parameters build.

## Components

| Component | Description |
| --------- | ----------- |
| [configuration.py](./configuration.py) | `DecompositionTransformerConfiguration`: sizes and attention builders of a variant. |
| [encoder_layer.py](./encoder_layer.py) | `DecompositionEncoderLayer`: attention and feed-forward network, keeping the seasonal part. |
| [decoder_layer.py](./decoder_layer.py) | `DecompositionDecoderLayer`: self- and cross-attention and feed-forward network, emitting a trend increment. |
| [seasonal_layer_norm.py](./seasonal_layer_norm.py) | `SeasonalLayerNorm`: layer normalization with the temporal mean removed. |
| [sequence_encoder.py](./sequence_encoder.py) | `DecompositionTransformerSequenceEncoder`: embedding and encoder layers. |
| [forecast_network.py](./forecast_network.py) | `DecompositionTransformerForecastNetwork`: encoder-decoder forecast of seasonal part and trend. |
