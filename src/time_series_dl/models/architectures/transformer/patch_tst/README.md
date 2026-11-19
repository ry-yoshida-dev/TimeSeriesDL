# patch_tst

## Overview

PatchTST (Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023): each
channel is cut into patches that a shared Transformer encoder reads independently of the other channels. Only
supervised training is provided, without the self-supervised pretraining of the paper.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `PatchTSTParameters`: patching, Transformer sizes, dropout and normalization. |
| [token_normalization.py](./token_normalization.py) | `TokenNormalization`: batch or layer normalization of the tokens. |
| [token_norm.py](./token_norm.py) | `PatchTSTTokenNorm`: the chosen normalization applied to token sequences. |
| [encoder_layer.py](./encoder_layer.py) | `PatchTSTEncoderLayer`: self-attention and feed-forward network. |
| [encoder.py](./encoder.py) | `PatchTSTEncoder`: padding, patching, embedding and encoder layers per channel. |
| [forecast_network.py](./forecast_network.py) | `PatchTSTForecastNetwork`: flattened patch features of each target channel to its horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `PatchTSTReconstructionNetwork`: flattened patch features of each channel to its sequence. |
| [summary_network.py](./summary_network.py) | `PatchTSTSummaryNetwork`: flattened patch features of all channels to the output vector. |
