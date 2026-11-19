# imputation

## Overview

Imputation of missing values in equal-length sequences with any model that builds a reconstruction network, trained
by hiding observed values at random.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `ImputerSettings`: model, training, mask ratio, loss and scaling. |
| [imputer.py](./imputer.py) | `Imputer`: `fit`, `impute`, `evaluate`, `save` and `load`. |
| [masked_reconstruction_step.py](./masked_reconstruction_step.py) | `MaskedReconstructionStep`: point loss on the hidden values of a batch. |
| [metrics.py](./metrics.py) | `ImputationMetrics`: MAE, MSE and RMSE of the hidden values. |
