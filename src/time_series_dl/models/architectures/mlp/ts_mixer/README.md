# ts_mixer

## Overview

TSMixer (Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting", TMLR 2023): blocks of
time-mixing and feature-mixing MLPs followed by a linear map along time, for every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TSMixerParameters`: blocks, hidden size, dropout, normalization and summary pooling. |
| [normalization.py](./normalization.py) | `TSMixerNormalization`: batch or layer normalization. |
| [batch_normalization.py](./batch_normalization.py) | `TSMixerBatchNormalization`: per-channel batch normalization of time-first sequences. |
| [mixer_block.py](./mixer_block.py) | `TSMixerBlock`: residual time-mixing and feature-mixing MLPs. |
| [forecast_network.py](./forecast_network.py) | `TSMixerForecastNetwork`: mixes the window, then maps it to the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `TSMixerReconstructionNetwork`: mixes the window, then maps it onto itself. |
| [summary_network.py](./summary_network.py) | `TSMixerSummaryNetwork`: mixes the window, then pools it to the output vector. |
