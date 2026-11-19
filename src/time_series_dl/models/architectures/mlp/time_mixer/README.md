# time_mixer

## Overview

TimeMixer (Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting", ICLR 2024): the
window is average-pooled into several scales, past decomposable mixing blocks mix the seasonal parts bottom-up and the
trends top-down across scales, and future multipredictor mixing sums the horizon predicted from every scale.

Following Time-Series-Library, forecasting and reconstruction encode each channel independently, and summaries embed
every channel into shared tokens. Unlike Time-Series-Library, every block adds its feed-forward output to its input as
in the paper (the library does so only for independent channels), the decomposition is the moving average only, time
features are not embedded, forecasts encode only the target channels, the reversible instance normalization of each
scale undoes only the parameters in the scale of the values, and reconstructions are not normalized.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimeMixerParameters`: sizes, scales, moving average, dropout and normalization. |
| [token_embedding.py](./token_embedding.py) | `TimeMixerTokenEmbedding`: circular convolution embedding of each time step. |
| [season_mixing.py](./season_mixing.py) | `TimeMixerSeasonMixing`: bottom-up mixing of the seasonal parts. |
| [trend_mixing.py](./trend_mixing.py) | `TimeMixerTrendMixing`: top-down mixing of the trends. |
| [decomposable_mixing.py](./decomposable_mixing.py) | `TimeMixerDecomposableMixing`: past decomposable mixing block. |
| [encoder.py](./encoder.py) | `TimeMixerEncoder`: down-sampling, embedding and mixing blocks. |
| [forecast_network.py](./forecast_network.py) | `TimeMixerForecastNetwork`: every scale of each target channel to the horizon, summed. |
| [reconstruction_network.py](./reconstruction_network.py) | `TimeMixerReconstructionNetwork`: the finest scale of each channel back to its values. |
| [summary_network.py](./summary_network.py) | `TimeMixerSummaryNetwork`: the flattened finest scale to the output vector. |
