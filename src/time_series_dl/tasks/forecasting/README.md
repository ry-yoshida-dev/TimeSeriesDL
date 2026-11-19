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

Forecasts are probabilistic with every objective but `PointObjective`, and `num_samples` adds sample paths to any
forecast. Generative models (TimeGrad, CSDI) are trained with their own loss, draw the sample paths with
`SampleObjective`, and derive the point forecast and the quantiles from them; `LowRankGaussianObjective` models the
dependence between target channels. `evaluate` reports the CRPS and the coverage of the forecast quantiles, and
`ConformalForecaster` calibrates intervals of any fitted forecaster on held-out series, adapting them online.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `ForecasterSettings`: model, window, objective, target channels, scaling of series and covariates, covariate adapter and training. |
| [forecaster.py](./forecaster.py) | `Forecaster`: `fit`, `predict`, `forecast`, `evaluate`, `save` and `load`, with or without covariates. |
| [window.py](./window.py) | `ForecastWindow`: input length, horizon and stride. |
| [objective.py](./objective.py) | `ForecastObjective`: predicted distribution of each value, its loss and its samples. |
| [objectives/](./objectives/README.md) | Point, quantile, Gaussian, Student's t, Laplace, low-rank multivariate Gaussian and sample objectives. |
| [sample_point_estimate.py](./sample_point_estimate.py) | `SamplePointEstimate`: mean or median of the samples as the point forecast. |
| [forecast_step.py](./forecast_step.py) | `ForecastStep`: parameters, samples and loss of a batch, with instance normalization of the past values. |
| [normalized_window.py](./normalized_window.py) | `NormalizedWindow`: a batch as the network reads it, with the statistics that undo the normalization. |
| [instance_normalization.py](./instance_normalization.py) | `InstanceNormalization`: per-window mean and standard deviation. |
| [prediction.py](./prediction.py) | `ForecastPrediction`: point forecasts, quantiles and sample paths. |
| [backtest.py](./backtest.py) | `ForecastBacktest`: forecasts of every window of some series with their true continuations. |
| [metrics.py](./metrics.py) | `ForecastMetrics`: MAE, MSE, RMSE, MAPE, sMAPE, quantile loss, CRPS and coverage. |
| [conformal/](./conformal/README.md) | `ConformalForecaster`: split conformal intervals and adaptive conformal inference. |

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

```python
from time_series_dl.models import CSDIParameters
from time_series_dl.tasks.forecasting import ConformalSettings, ForecasterSettings, SampleObjective

forecaster = ForecasterSettings(model=CSDIParameters(), window=window, objective=SampleObjective()).build()
forecaster.fit(training_series)
prediction = forecaster.forecast(series, num_samples=100)  # prediction.samples: (1, 100, horizon, channels)
metrics = forecaster.evaluate(test_series)  # metrics.crps, metrics.interval_coverage
conformal = ConformalSettings(coverage_levels=(0.9,)).build(forecaster)
conformal.calibrate(calibration_series)
interval = conformal.forecast(series)  # interval.get_quantile(0.05), interval.get_quantile(0.95)
```
