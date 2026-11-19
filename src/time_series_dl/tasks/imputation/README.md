# imputation

## Overview

Imputation of missing values in equal-length sequences, trained by hiding observed values at random. Models that read
the observed mask (SAITS, BRITS, GRU-D) build a masked reconstruction network and may add their own training loss;
any other model that builds a reconstruction network sees the missing values as zeros.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `ImputerSettings`: model, training, mask ratio, loss and scaling. |
| [imputer.py](./imputer.py) | `Imputer`: `fit`, `impute`, `evaluate`, `save` and `load`. |
| [masked_reconstruction_step.py](./masked_reconstruction_step.py) | `MaskedReconstructionStep`: weighted loss on the hidden values of a batch plus the network's own loss. |
| [zero_filled_network.py](./zero_filled_network.py) | `ZeroFilledNetwork`: a reconstruction network that does not read the mask, as a masked one. |
| [metrics.py](./metrics.py) | `ImputationMetrics`: MAE, MSE and RMSE of the hidden values. |
