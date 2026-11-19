# sparse_tsf

## Overview

SparseTSF (Lin et al., "SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters", ICML 2024): a
sliding aggregation (a zero-padded convolution with a residual connection) smooths each channel, the window is split
into periods, and one linear map shared by every phase of the period forecasts the corresponding steps of the future
periods from the same phase of the past periods.

Unlike the official code, the mean of the window is not subtracted and added back (the forecasting task normalizes
each window), windows and horizons need not be multiples of the period (the window is zero-padded at the front so that
it ends with a whole period, and the forecast keeps the first `horizon` steps of the whole future periods), each
distribution parameter has its own outputs of the linear map, and the MLP variant of the journal extension is not
included.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `SparseTSFParameters`: period length. |
| [forecast_network.py](./forecast_network.py) | `SparseTSFForecastNetwork`: aggregation, down-sampling by period and the shared linear map. |
