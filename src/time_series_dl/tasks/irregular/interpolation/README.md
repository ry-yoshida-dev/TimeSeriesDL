# interpolation

## Overview

Prediction of irregularly sampled sequences at any query times, between their observations (interpolation) or after
them (forecasting), with any model that builds an irregular interpolation network. Training hides part of each
sequence from the network, either random observations or the later part of its time range, and minimizes the
network's own loss of the hidden values, such as the negative evidence lower bound of mTAN and the Latent ODE.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `IrregularInterpolatorSettings`: model, training, scaling and context selection. |
| [interpolator.py](./interpolator.py) | `IrregularInterpolator`: `fit`, `predict`, `evaluate`, `save` and `load`. |
| [context_selection.py](./context_selection.py) | `ContextSelection`: random observations (interpolation) or earlier times (forecasting) as context. |
| [interpolation_step.py](./interpolation_step.py) | `IrregularInterpolationStep`: context and targets of a batch, and the network's loss. |
| [metrics.py](./metrics.py) | `InterpolationMetrics`: MAE, MSE and RMSE of the observed targets. |
