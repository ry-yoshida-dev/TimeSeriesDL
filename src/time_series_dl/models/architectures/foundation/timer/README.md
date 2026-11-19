# timer

## Overview

Timer (Liu et al., "Timer: Generative Pre-trained Transformers Are Large Time Series Models", ICML 2024): a
decoder-only Transformer pretrained on 260B time points that reads non-overlapping patches of 96 values and predicts
the next patch, generating longer horizons patch by patch. `TimerParameters.pretrained(TimerSize.BASE)` loads the
published `thuml/timer-base-84m` checkpoint (Apache-2.0) at a pinned revision; its outputs match the official
`generate` of the repository's `modeling_timer.py` (transformers 4.40.1) to 1.2e-4 on values around 200. Each target
channel is forecast as a separate univariate series, and only point forecasts are available (`PointObjective`).

The network normalizes each series itself, as the official generation does: the window, cut to its last whole
number of patches, by its mean and sample standard deviation, and each generated patch by its own statistics before
it is fed back (the effect of the official key-value cache, which normalizes only the newly appended values). The
forecast is therefore unchanged by the task's scaling and instance normalization, up to rounding;
`NoScalingParameters()` with `is_instance_normalized=False` is the simplest setting. Fine-tuning with `fit` trains
on the forecast horizon with the task's objective, through the patch-by-patch generation, instead of the official
next-patch loss over every position.

Differences from the official code: a standard deviation of zero is replaced by one (the original divides by zero),
the attention dropout is applied only while training (the original applies its configured probability, 0 for the
published checkpoint, also at inference), and only one output head is supported (the published checkpoint has one).

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimerParameters`: patch lengths, Transformer sizes and published weights. |
| [size.py](./size.py) | `TimerSize`: the published checkpoint and its pinned revision. |
| [rotary_embedding.py](./rotary_embedding.py) | `TimerRotaryEmbedding`: rotary position embedding of queries and keys. |
| [attention.py](./attention.py) | `TimerAttention`: causal multi-head self-attention. |
| [feed_forward.py](./feed_forward.py) | `TimerFeedForward`: SiLU-gated feed-forward network. |
| [decoder_layer.py](./decoder_layer.py) | `TimerDecoderLayer`: post-normalized attention and feed-forward network. |
| [model.py](./model.py) | `TimerModel`: patch embedding, decoder layers and the next-patch head. |
| [forecast_network.py](./forecast_network.py) | `TimerForecastNetwork`: normalization and patch-by-patch generation of each target channel. |
| [official_names.py](./official_names.py) | `TimerOfficialNames`: renamings of the published parameter names. |

## Examples

```python
from data_scaler import NoScalingParameters

from time_series_dl.models.architectures.foundation.timer import TimerParameters, TimerSize
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow

forecaster = ForecasterSettings(
    model=TimerParameters.pretrained(TimerSize.BASE),
    window=ForecastWindow(input_length=672, horizon=96),
    scaling=NoScalingParameters(),
    is_instance_normalized=False,
).build()
forecaster.prepare_zero_shot(series)  # downloads the weights; needs the `pretrained` extra
prediction = forecaster.forecast(series)
```
