# xlstm_time

## Overview

xLSTMTime (Alharthi and Mahmood, "xLSTMTime: Long-term Time Series Forecasting With xLSTM", 2024): the window is
decomposed into trend and seasonal parts, each mapped linearly to the horizon; the sum is embedded per channel,
batch normalized, and refined by an xLSTM stack that reads the channels as its sequence, before a linear layer maps
each channel back to the horizon. Forecasting only.

Differences from the paper: the reversible instance normalization is applied by the forecasting task, not by the
network; the output layer predicts every distribution parameter; and the paper does not state the number of blocks,
so the default is a single sLSTM block (the paper's choice for the smaller datasets; an empty
`slstm_block_indices` gives its mLSTM variant). Non-target channels are read as inputs and only the target channels
are returned.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `XLSTMTimeParameters`: moving average, model size and xLSTM stack. |
| [forecast_network.py](./forecast_network.py) | `XLSTMTimeForecastNetwork`: decomposition, linear forecasts and the stack over channels. |
