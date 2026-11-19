# forecasting

## Overview

Forecasting of the next `horizon` steps of multivariate series from their last `input_length` steps, with any model
that builds a forecast network. Series are scaled per channel, and each window is optionally normalized by its own
statistics (RevIN); forecasts are returned in the original scale.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `ForecasterSettings`: model, window, objective, target channels, scaling and training. |
| [forecaster.py](./forecaster.py) | `Forecaster`: `fit`, `predict`, `forecast`, `evaluate`, `save` and `load`. |
| [window.py](./window.py) | `ForecastWindow`: input length, horizon and stride. |
| [objective.py](./objective.py) | `ForecastObjective`: predicted distribution of each value and its loss. |
| [objectives/](./objectives/README.md) | Point, quantile, Gaussian, Student's t and Laplace objectives. |
| [forecast_step.py](./forecast_step.py) | `ForecastStep`: parameters and loss of a batch, with instance normalization. |
| [instance_normalization.py](./instance_normalization.py) | `InstanceNormalization`: per-window mean and standard deviation. |
| [prediction.py](./prediction.py) | `ForecastPrediction`: point forecasts and quantiles. |
| [metrics.py](./metrics.py) | `ForecastMetrics`: MAE, MSE, RMSE, MAPE, sMAPE and quantile loss. |
