# ttm

## Overview

TinyTimeMixer, TTM (Ekambaram et al., "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot
Forecasting of Multivariate Time Series", NeurIPS 2024): a pretrained TSMixer of about 1M parameters. Each channel is
standardized by its own mean and standard deviation, cut into patches, mixed across patches and features by a
backbone with adaptive patching (each level splits every patch into 2^i finer patches) and by a smaller decoder,
and a linear head forecasts a fixed number of values at once. `TTMParameters.pretrained(variant)` loads one of the
twelve r2 checkpoints of `ibm-granite/granite-timeseries-ttm-r2` (Apache-2.0; contexts 512, 1024 and 1536, predictions
96, 192, 336 and 720, each a pinned branch commit); outputs match `TinyTimeMixerForPrediction` of granite-tsfm
0.3.x to 1.2e-4 on values around 200 for the 512-96 and 1024-192 variants, including shorter (zero-padded) and
longer (cut) windows. Each target channel is forecast as a separate series, as the channel-independent checkpoints
do, and only point forecasts are available (`PointObjective`); the horizon may not exceed the variant's prediction
length, of which the first `horizon` values are returned.

The network standardizes each window itself; its scale is `sqrt(variance + scale_epsilon)`, so the forecasts are
the official ones with `NoScalingParameters()` and `is_instance_normalized=False`, and differ only through that
epsilon otherwise. Windows shorter than `context_length` are padded with zeros on the left that count in the
standardization, as in the original, so a window of the full context length is preferable. Fine-tuning with `fit`
trains every layer with the task's objective.

Only what the published r2 checkpoints use is implemented: no channel mixing, exogenous forecast channel mixing,
static categorical features, resolution prefix tuning, patch self-attention, raw-residual decoder, register or FFT
tokens, batch normalization, distribution or multi-quantile heads.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TTMParameters`: context and prediction lengths, patching, mixer sizes and published weights. |
| [variant.py](./variant.py) | `TTMVariant`: the published r2 checkpoints and their pinned revisions. |
| [mixer_configuration.py](./mixer_configuration.py) | `TTMMixerConfiguration`: sizes of the backbone or the decoder. |
| [gated_attention.py](./gated_attention.py) | `TTMGatedAttention`: softmax gating of the features. |
| [perceptron.py](./perceptron.py) | `TTMPerceptron`: the two-layer GELU perceptron of the mixers. |
| [patch_mixer.py](./patch_mixer.py) | `TTMPatchMixer`: mixing across patches. |
| [feature_mixer.py](./feature_mixer.py) | `TTMFeatureMixer`: mixing across features. |
| [mixer_layer.py](./mixer_layer.py) | `TTMMixerLayer`: patch and feature mixing. |
| [mixer_block.py](./mixer_block.py) | `TTMMixerBlock`: mixer layers at one adaptive patching level. |
| [mixer_stack.py](./mixer_stack.py) | `TTMMixerStack`: the blocks of all levels. |
| [model.py](./model.py) | `TTMModel`: standardization, patching, backbone, decoder and head of one channel. |
| [forecast_network.py](./forecast_network.py) | `TTMForecastNetwork`: the model applied to each target channel. |
| [official_names.py](./official_names.py) | `TTMOfficialNames`: renamings of the published parameter names. |

## Examples

```python
from data_scaler import NoScalingParameters

from time_series_dl.models.architectures.foundation.ttm import TTMParameters, TTMVariant
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow

forecaster = ForecasterSettings(
    model=TTMParameters.pretrained(TTMVariant.CONTEXT_512_PREDICTION_96),
    window=ForecastWindow(input_length=512, horizon=96),
    scaling=NoScalingParameters(),
    is_instance_normalized=False,
).build()
forecaster.prepare_zero_shot(series)  # downloads the weights; needs the `pretrained` extra
prediction = forecaster.forecast(series)
```
