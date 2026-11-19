# data

## Overview

Validation, windowing and scaling of the arrays that tasks receive.

## Components

| Component | Description |
| --------- | ----------- |
| [series_collection.py](./series_collection.py) | `SeriesCollection`: one or more series of shape (length, channels) that share their channels. |
| [sequence_array.py](./sequence_array.py) | `SequenceArray`: a batch of equal-length sequences of shape (num_sequences, length, channels). |
| [sliding_windows.py](./sliding_windows.py) | `SlidingWindows`: fixed-length windows of a series at a fixed stride. |
| [channel_scaler.py](./channel_scaler.py) | `ChannelScaler`: a `data_scaler` strategy applied per channel over all time steps. |
| [covariates/](./covariates/README.md) | Known and static covariates of forecasts: validation, windowing, scaling and calendar features. |
| [irregular/](./irregular/README.md) | Irregularly sampled sequences: validation, padding with masks, and scaling of observed values. |
