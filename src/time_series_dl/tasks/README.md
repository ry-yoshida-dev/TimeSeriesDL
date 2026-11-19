# tasks

## Overview

One interface per time series task. Each task is built from a settings dataclass that takes a model's parameters,
and trains with the shared training loop.

## Components

| Component | Description |
| --------- | ----------- |
| [forecasting/](./forecasting/README.md) | `Forecaster`: point, quantile or Gaussian forecasts of the next horizon. |
| [classification/](./classification/README.md) | `SequenceClassifier`: one class per sequence. |
| [regression/](./regression/README.md) | `SequenceRegressor`: one or more values per sequence. |
| [imputation/](./imputation/README.md) | `Imputer`: fills missing values. |
| [anomaly_detection/](./anomaly_detection/README.md) | `AnomalyDetector`: reconstruction-error scores of every time step. |
