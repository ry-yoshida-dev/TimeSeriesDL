# tests

## Overview

Unit tests of every task, model, objective and utility. They run on the CPU in seconds with small networks and
synthetic series.

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Fixtures: fast training settings and a synthetic multivariate series. |
| [network_checks.py](./network_checks.py) | Shape and gradient checks of the networks of every model. |
| `test_model_*.py` | Network checks and model-specific behavior of each model. |
| [test_state_space_components.py](./test_state_space_components.py) | HiPPO memories, kernels, scans and convolutions of the state space models against their recurrences. |
| [test_reference_models.py](./test_reference_models.py) | Network checks of MLP, RNN, TCN, DLinear and NLinear. |
| [test_model_settings.py](./test_model_settings.py) | `ModelSettings` and the YAML presets of every model. |
| [test_forecaster.py](./test_forecaster.py) | Forecasting: objectives, target channels, scaling and validation. |
| [test_classifier.py](./test_classifier.py) | Classification of sequences. |
| [test_regressor.py](./test_regressor.py) | Regression of one or several outputs. |
| [test_imputer.py](./test_imputer.py) | Imputation of missing values. |
| [test_anomaly_detector.py](./test_anomaly_detector.py) | Anomaly scores, detection and metrics. |
| [test_objectives.py](./test_objectives.py) | Forecast objectives and metrics. |
| [test_losses.py](./test_losses.py) | Point and classification losses. |
| [test_training.py](./test_training.py) | Trainer, optimizers, schedules and early stopping. |
| [test_persistence.py](./test_persistence.py) | Saving and loading every task and model, and rejecting invalid checkpoints. |
| [test_data.py](./test_data.py) | Windows, series validation and channel scaling. |
