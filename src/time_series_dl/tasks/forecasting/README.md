# forecasting

## Overview

Forecasting of the next `horizon` steps of multivariate series from their last `input_length` steps, with any model
that builds a forecast network. Series are scaled per channel, and each window is optionally normalized by its own
statistics (RevIN); forecasts are returned in the original scale.

Covariates are optional. Channels that are not targets are past-observed covariates. `SeriesCovariates` add known
covariates, which cover the past window and the horizon (calendar features from `CalendarFeatures`, planned
promotions), and static numeric and categorical features of each series. Covariates are scaled with statistics of the
training covariates and never normalized by the window; with instance normalization, covariate networks also read
each window's target location and log scale as static features. TiDE, Temporal Fusion Transformer and DeepAR read
covariates as in their papers; every other model reads them through the generic `CovariateAdapterNetwork`, which
appends the past known covariates to the input channels and corrects each horizon step from its known covariates and
the static features. A forecaster fitted with covariates needs them wherever it forecasts, and one fitted without them
rejects them, with `CovariateMismatchError`.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `ForecasterSettings`: model, window, objective, target channels, scaling of series and covariates, covariate adapter and training. |
| [forecaster.py](./forecaster.py) | `Forecaster`: `fit`, `predict`, `forecast`, `evaluate`, `save` and `load`, with or without covariates. |
| [window.py](./window.py) | `ForecastWindow`: input length, horizon and stride. |
| [objective.py](./objective.py) | `ForecastObjective`: predicted distribution of each value and its loss. |
| [objectives/](./objectives/README.md) | Point, quantile, Gaussian, Student's t and Laplace objectives. |
| [forecast_step.py](./forecast_step.py) | `ForecastStep`: parameters and loss of a batch, with instance normalization of the past values. |
| [instance_normalization.py](./instance_normalization.py) | `InstanceNormalization`: per-window mean and standard deviation. |
| [prediction.py](./prediction.py) | `ForecastPrediction`: point forecasts and quantiles. |
| [metrics.py](./metrics.py) | `ForecastMetrics`: MAE, MSE, RMSE, MAPE, sMAPE and quantile loss. |

## Examples

```python
import numpy as np

from time_series_dl.models import TiDEParameters
from time_series_dl.tasks.forecasting import CalendarFeatures, ForecasterSettings, ForecastWindow, SeriesCovariates

window = ForecastWindow(input_length=96, horizon=24)
calendar = CalendarFeatures()
# sales: (length, channels); promotions: (length + 24, 1), planned through the horizon; timestamps: (length + 24,).
known = np.concatenate([calendar.compute(timestamps), promotions], axis=1)
covariates = SeriesCovariates(known=known, static_categories=np.array([store_id]))

forecaster = ForecasterSettings(model=TiDEParameters(), window=window, target_channels=(0,)).build()
forecaster.fit(sales, covariates=SeriesCovariates(known=known[:-24], static_categories=np.array([store_id])))
prediction = forecaster.forecast(sales, covariates)  # known covariates extend 24 steps past the series
```
