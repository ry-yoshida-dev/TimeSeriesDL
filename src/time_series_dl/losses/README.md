# losses

## Overview

Point losses shared by the tasks that predict values: forecasting, regression, imputation and anomaly detection.

## Components

| Component | Description |
| --------- | ----------- |
| [point_loss.py](./point_loss.py) | `PointLoss`: loss of each value, with mean and masked mean. |
| [point_losses/](./point_losses/README.md) | MSE, MAE, Huber, log-cosh and pinball losses. |
