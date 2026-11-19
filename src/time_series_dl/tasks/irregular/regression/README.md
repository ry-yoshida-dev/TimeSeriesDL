# regression

## Overview

Regression of one or more values from irregularly sampled sequences with any model that builds an irregular summary
network. Losses are the point losses of [losses](../../../losses/README.md), and metrics are the types of
`classreg.regression`, one per output.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `IrregularRegressorSettings`: model, training, loss and the scaling of values and targets. |
| [regressor.py](./regressor.py) | `IrregularSequenceRegressor`: `fit`, `predict`, `evaluate`, `save` and `load`. |
| [regression_step.py](./regression_step.py) | `IrregularRegressionStep`: training loss of a batch, with the network's own terms. |
