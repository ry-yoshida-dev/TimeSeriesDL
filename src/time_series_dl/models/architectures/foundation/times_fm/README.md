# times_fm

## Overview

TimesFM 2.0 (Das et al., "A decoder-only foundation model for time-series forecasting", ICML 2024): a pretrained
decoder-only transformer over patches of 32 values. Each series is normalized by the mean and standard deviation of
its first patch with at least three observed values; every patch is embedded with its padding mask by a residual
block, a frequency embedding is added, and 50 causal layers with per-dimension query scaling follow. The last patch
is projected to the mean and the 0.1, ..., 0.9 quantiles of the next 128 values; longer horizons are decoded
autoregressively by appending the mean forecast to the context, which keeps its last 2048 values.

`TimesFMParameters.pretrained(TimesFMRelease.V2_0_500M)` loads `google/timesfm-2.0-500m-pytorch` (Apache-2.0, about
2 GB) at a pinned revision. The network forecasts each target channel as a separate series and ignores the other
channels; with one parameter per value it outputs the mean head (the official point forecast), and with
`QuantileObjective((0.1, 0.2, ..., 0.9))` the quantile heads, so it forecasts zero-shot through
`Forecaster.prepare_zero_shot` and fine-tunes through `fit`. The model normalizes each series itself; use
`NoScalingParameters()` and `is_instance_normalized=False` to feed it raw values (its normalization is affine
invariant, so other settings change the forecasts only through numerical precision).

Differences from the official code: the published PyTorch weights are those of the Hugging Face `transformers` port
(`TimesFmModelForPrediction`), whose names `TimesFMOfficialNames` maps. Contexts are left-padded to whole patches
instead of to the full context length, which gives the same forecasts because padded patches are masked. Decoding
beyond 128 values follows the original `timesfm` package (the `transformers` port decodes only one block, and its
loop would misalign the padding mask in later blocks). Missing values (NaN) are treated as padding. The positional
embedding of TimesFM 1.0, the covariate regression of the `timesfm` package, `truncate_negative` and the
trend/residual decomposition option are not implemented. Quantile objectives sort the quantile heads, which the
official code does not.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimesFMParameters`: sizes, patching, quantile levels, frequency and published weights. |
| [release.py](./release.py) | `TimesFMRelease`: published checkpoints with their pinned revisions. |
| [frequency.py](./frequency.py) | `TimesFMFrequency`: frequency category of the series and its embedding row. |
| [residual_block.py](./residual_block.py) | `TimesFMResidualBlock`: SiLU multilayer perceptron with a linear shortcut. |
| [attention.py](./attention.py) | `TimesFMAttention`: self-attention with learned per-dimension query scaling. |
| [feed_forward.py](./feed_forward.py) | `TimesFMFeedForward`: normalized ReLU feed-forward network with padding-masked residual. |
| [decoder_layer.py](./decoder_layer.py) | `TimesFMDecoderLayer`: causal self-attention and feed-forward network. |
| [model.py](./model.py) | `TimesFMModel`: normalization, patch embedding, decoder layers and the output block of the last patch. |
| [forecast_network.py](./forecast_network.py) | `TimesFMForecastNetwork`: autoregressive forecasts of each target channel. |
| [official_names.py](./official_names.py) | `TimesFMOfficialNames`: renaming of the published parameter names. |
