# chronos_bolt

## Overview

Chronos-Bolt (Ansari et al., "Chronos: Learning the Language of Time Series", TMLR 2024; Chronos-Bolt release,
2024): a pretrained T5 encoder-decoder over patches. Each series is standardized by the mean and standard deviation
of its observed context, cut into patches of 16 values that a residual block embeds together with their
observation mask, and encoded with a trailing register token; one decoder step turns the encoder states into nine
quantiles of the next 64 values at once. Every target channel is forecast as a separate univariate series.

`ChronosBoltParameters.pretrained(ChronosBoltSize.SMALL)` loads `amazon/chronos-bolt-{tiny,mini,small,base}`
(Apache-2.0) at a pinned revision when the network is built (optional `pretrained` extra);
`Forecaster.prepare_zero_shot` forecasts with it without training, and `fit` fine-tunes it with the task's
objective. Build it for `PointObjective` (the median) or `QuantileObjective(quantile_levels)`. The model's own
standardization is affine invariant, so the task's scaling and instance normalization change its forecasts only
within floating-point precision; `NoScalingParameters()` with `is_instance_normalized=False` feeds it exactly the
values it was trained on.

Differences from chronos-forecasting: horizons beyond 64 values follow the pipeline's heuristic (every quantile is
appended to the context as a path, and the quantiles over all paths of the next block are taken) inside the
network, so fine-tuning back-propagates through it. Fine-tuning uses the task's loss, not the official quantile loss
scaled by two and summed over levels. Missing values (NaN) are handled as in the original, although the task never
passes any.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ChronosBoltParameters`: T5 sizes, patching, quantile levels and published weights. |
| [size.py](./size.py) | `ChronosBoltSize`: the published checkpoints and their pinned revisions. |
| [residual_block.py](./residual_block.py) | `ChronosBoltResidualBlock`: residual MLP of the patch embedding and the output projection. |
| [model.py](./model.py) | `ChronosBoltModel`: standardization, patching, T5 encoder-decoder and quantile projection. |
| [forecast_network.py](./forecast_network.py) | `ChronosBoltForecastNetwork`: each target channel as a univariate series, in blocks of 64 values. |
| [official_names.py](./official_names.py) | `ChronosBoltOfficialNames`: the published parameter names. |
