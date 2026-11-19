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
| [anomaly_detection/](./anomaly_detection/README.md) | `AnomalyDetector`: reconstruction-error, forecast-error or model-specific scores of every time step. |
| [representation_learning/](./representation_learning/README.md) | `RepresentationLearner`: self-supervised pretraining of an encoder and its representations. |
| [irregular/](./irregular/README.md) | `IrregularSequenceClassifier`, `IrregularSequenceRegressor` and `IrregularInterpolator` for irregularly sampled sequences. |
| [generation/](./generation/README.md) | `SequenceGenerator`: synthetic sequences from a generative model, with discriminative, predictive and distribution metrics. |
