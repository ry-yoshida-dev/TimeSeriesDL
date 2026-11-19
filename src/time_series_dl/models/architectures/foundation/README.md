# foundation

## Overview

Time series foundation models: pretrained networks whose officially published weights load from the Hugging Face
Hub (optional `pretrained` extra, see [components/pretrained/](../../components/pretrained/README.md)). Each
parameters class has a `pretrained` constructor for its published checkpoints, pinned to a commit; built with
weights, a forecast network forecasts without training through `Forecaster.prepare_zero_shot`, and `fit` fine-tunes
it. The models scale each series themselves, so use `NoScalingParameters()` and `is_instance_normalized=False`.
Built without weights, the same architectures train from scratch. Check each checkpoint's license: the Moirai
weights are for non-commercial use only.

A saved task reloads its weights from its checkpoint without the Hugging Face Hub.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation / anomaly detection | Weights license |
| --------- | ---------- | :---------: | :-------------------------: | :----------------------------: | --------------- |
| [Chronos (Ansari et al., 2024)](./chronos/README.md) | `ChronosParameters` | ✓ zero-shot, samples | | | Apache-2.0 |
| [Chronos-Bolt (Ansari et al., 2024)](./chronos_bolt/README.md) | `ChronosBoltParameters` | ✓ zero-shot, quantiles | | | Apache-2.0 |
| [TimesFM 2.0 (Das et al., 2024)](./times_fm/README.md) | `TimesFMParameters` | ✓ zero-shot, quantiles | | | Apache-2.0 |
| [Moirai 1.1-R (Woo et al., 2024)](./moirai/README.md) | `MoiraiParameters` | ✓ zero-shot, samples | | | CC-BY-NC-4.0 |
| [MOMENT (Goswami et al., 2024)](./moment/README.md) | `MOMENTParameters` | ✓ fine-tuned head | ✓ fine-tuned head | ✓ zero-shot | MIT |
| [Timer (Liu et al., 2024)](./timer/README.md) | `TimerParameters` | ✓ zero-shot, point | | | Apache-2.0 |
| [TTM (Ekambaram et al., 2024)](./ttm/README.md) | `TTMParameters` | ✓ zero-shot, point | | | Apache-2.0 |

## Examples

```python
from data_scaler import NoScalingParameters

from time_series_dl.models.architectures.foundation import ChronosBoltParameters, ChronosBoltSize
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow, QuantileObjective

model = ChronosBoltParameters.pretrained(ChronosBoltSize.SMALL)
forecaster = ForecasterSettings(
    model=model,
    window=ForecastWindow(input_length=512, horizon=64),
    objective=QuantileObjective(model.quantile_levels),
    scaling=NoScalingParameters(),
    is_instance_normalized=False,
).build()
forecaster.prepare_zero_shot(series)  # downloads the weights once; no training
prediction = forecaster.forecast(series)
forecaster.fit(train_series)  # fine-tunes from the published weights
```
