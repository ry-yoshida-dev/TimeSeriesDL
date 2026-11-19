# regression

## Overview

Regression of one or more values from whole sequences with any model that builds a summary network. Metrics are
the types of `classreg.regression`, one per output.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `RegressorSettings`: model, training, loss and the scaling of sequences and targets. |
| [regressor.py](./regressor.py) | `SequenceRegressor`: `fit`, `predict`, `evaluate`, `save` and `load`. |
| [regression_step.py](./regression_step.py) | `RegressionStep`: point loss of a batch. |
