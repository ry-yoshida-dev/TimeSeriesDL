# time_series_dl

## Overview

Deep learning models for time series behind one interface per task. Each task is built from a settings dataclass
that names a model's parameters; a model supports a task when it can build the network the task needs.

## Components

| Component | Description |
| --------- | ----------- |
| [tasks/](./tasks/README.md) | Forecasting, classification, regression, imputation and anomaly detection. |
| [models/](./models/README.md) | Parameters and networks of every model. |
| [config/](./config/README.md) | YAML presets of every model for `ModelSettings`. |
| [training/](./training/README.md) | `TrainingSettings` and the shared training loop. |
| [data/](./data/README.md) | Validation, windowing and per-channel scaling of input arrays. |
| [losses/](./losses/README.md) | `PointLoss`: MSE, MAE, Huber, log-cosh and pinball losses. |
| [persistence/](./persistence/README.md) | Checkpoint files of fitted tasks, loaded without pickle. |
| [runtime/](./runtime/README.md) | `Device` and batched inference. |
| [array_types.py](./array_types.py) | NumPy array aliases. |
