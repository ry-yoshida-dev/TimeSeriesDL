# cycle_net

## Overview

CycleNet (Lin et al., "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns", NeurIPS
2024): a learnable recurrent cycle of `cycle_length` values per channel is removed from each window at the phase of
every step, a linear or MLP backbone shared by the channels forecasts the residual, and the cycle at the phases of
the horizon is added back. Forecasting only.

CycleNet needs the phase of each step, so it forecasts only with known covariates (`CovariateForecastingModel`):
known covariate channel `phase_channel` holds the integer phase in `[0, cycle_length)` for the window and the
horizon, and the forecaster must not scale the covariates (`covariate_scaling=NoScalingParameters()`). Without
covariates, building the network raises `ValueError`. Other covariates, static features and past-observed channels
that are not targets are not read.

Differences from the official code:

- The cycle is read at the phase of every step from the covariate, instead of at `(start + step) % cycle_length`
  from one index per window; the two agree when the phase counts up by one per step. The official loader's index is
  the phase of the first horizon step, which only rotates the learned cycle.
- Instance normalization is applied by the forecasting task, as the official `use_revin`.
- The backbone predicts every distribution parameter; the cycle is added to those in the scale of the values.

The defaults follow the official ETTh1 scripts (`scripts/CycleNet/MLP-Input-96/etth1.sh` of ACAT-SCUT/CycleNet with
`run.py`'s defaults): a daily cycle of 24 hourly steps and the MLP backbone with 512 hidden units.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `CycleNetParameters`: cycle length, backbone, hidden size and phase channel. |
| [backbone.py](./backbone.py) | `CycleNetBackbone`: linear or MLP map from the residual window to the horizon. |
| [recurrent_cycle.py](./recurrent_cycle.py) | `CycleNetRecurrentCycle`: learnable cycle read at the phase of each step. |
| [covariate_forecast_network.py](./covariate_forecast_network.py) | `CycleNetCovariateForecastNetwork`: removes the cycle, forecasts the residual and adds the cycle back. |

## Examples

Hourly data with a weekly cycle: the phase is the hour of the week, counted from Monday 00:00 (1970-01-01, the
zero of `datetime64`, was a Thursday, 72 hours after a Monday).

```python
import numpy as np
from data_scaler import NoScalingParameters

from time_series_dl.data import SeriesCovariates
from time_series_dl.models.architectures.linear.cycle_net import CycleNetParameters
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow

timestamps = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[h]")
series = np.random.default_rng(0).normal(size=(timestamps.size, 1))
hour_of_week = (timestamps.astype(np.int64) + 72) % 168
forecaster = ForecasterSettings(
    model=CycleNetParameters(cycle_length=168),
    window=ForecastWindow(input_length=96, horizon=24),
    covariate_scaling=NoScalingParameters(),
).build()
forecaster.fit(series, covariates=SeriesCovariates(known=hour_of_week[:, np.newaxis].astype(np.float64)))
```
