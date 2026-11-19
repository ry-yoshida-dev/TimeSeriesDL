# moirai

## Overview

Moirai (Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", ICML 2024), release 1.1-R:
a masked Transformer encoder pretrained on LOTSA. Every variate is standardized by its own past values and cut into
patches (one of several patch sizes); the patches of all variates and masked placeholder tokens for the horizon form
one sequence, read by any-variate attention (rotary embedding of the time index, learned bias for same and different
variates). Every forecast token gets a mixture of a Student's t, a narrow normal, a negative binomial and a
log-normal distribution per value, and forecasts are sampled from it. Moirai is natively multivariate: the target
channels are forecast together and the other channels are read as past-only covariates.

**License: the published weights `Salesforce/moirai-1.1-R-{small,base,large}` are licensed CC-BY-NC-4.0 and may be
used for non-commercial purposes only.** The original code (uni2ts) is Apache-2.0.

`MoiraiParameters.pretrained(MoiraiSize.SMALL)` downloads a checkpoint at a pinned revision (optional `pretrained`
extra). The network is a `GenerativeForecastNetwork`: use `SampleObjective` (the original draws 100 samples).
Fine-tuning minimizes the mixture negative log-likelihood of the normalized horizon. The model standardizes every
variate itself, which is invariant to affine transformations up to the variance floor; `NoScalingParameters()` with
`is_instance_normalized=False` feeds it exactly the values it was trained on.

Differences from uni2ts:

- The context is the whole input window with a fixed `patch_size`. With `patch_size=None` the original's
  `patch_size="auto"` is reproduced: the last `input_length - horizon` steps are the context and each patch size is
  scored on the window's last `horizon` steps, so the input window must be `context_length + horizon` long. The
  fine-tuning loss then averages over every patch size, where uni2ts fine-tunes with one.
- The loss leaves out the log-scale term of the density in the input scale, which does not depend on the weights.
- Samples come from PyTorch's global generator, seeded from the given generator, so they differ from uni2ts'
  samples for the same seed; the distributions themselves agree.
- Attention dropout is applied only while training (uni2ts applies it always; it is 0 in the published models).
- Statistics are computed in single instead of double precision on Apple MPS, which has no double precision.

Parity with uni2ts 2.0.0 on `moirai-1.1-R-small` (500-step context, 40-step horizon, two targets and one covariate):
mixture parameters and log densities agree to 7e-6, the patch-size validation losses to 1e-6.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MoiraiParameters`: sizes, patch sizes, scaling and published weights. |
| [size.py](./size.py) | `MoiraiSize`: the published checkpoints and their pinned revisions. |
| [official_names.py](./official_names.py) | `MoiraiOfficialNames`: translation of the published parameter names. |
| [forecast_network.py](./forecast_network.py) | `MoiraiForecastNetwork`: channels as variates, sampling, loss and patch size selection. |
| [model.py](./model.py) | `MoiraiModel`: standardization, patching, token sequence and encoder. |
| [forecast_distribution.py](./forecast_distribution.py) | `MoiraiForecastDistribution`: sampling and likelihood of the horizon. |
| [mixture.py](./mixture.py) | `MoiraiMixture`: the four-component mixture distribution. |
| [distribution_head.py](./distribution_head.py) | `MoiraiDistributionHead`: token representations to mixture parameters. |
| [input_projection.py](./input_projection.py) | `MoiraiInputProjection`: one patch embedding per patch size. |
| [output_projection.py](./output_projection.py) | `MoiraiOutputProjection`: one parameter projection per patch size. |
| [encoder.py](./encoder.py) | `MoiraiEncoder`: the encoder layers and the final normalization. |
| [encoder_layer.py](./encoder_layer.py) | `MoiraiEncoderLayer`: any-variate attention and gated feed-forward network. |
| [attention.py](./attention.py) | `MoiraiAttention`: attention with normalized queries and keys and a variate bias. |
| [rotary_embedding.py](./rotary_embedding.py) | `MoiraiRotaryEmbedding`: rotary embedding of the time index on half of each head. |
| [feed_forward.py](./feed_forward.py) | `MoiraiFeedForward`: SwiGLU feed-forward network. |

## Examples

```python
from data_scaler import NoScalingParameters

from time_series_dl.models.architectures.foundation.moirai import MoiraiParameters, MoiraiSize
from time_series_dl.tasks.forecasting import ForecasterSettings, ForecastWindow, SampleObjective

forecaster = ForecasterSettings(
    model=MoiraiParameters.pretrained(MoiraiSize.SMALL, patch_size=32),
    window=ForecastWindow(input_length=512, horizon=48),
    objective=SampleObjective(num_samples=100),
    scaling=NoScalingParameters(),
    is_instance_normalized=False,
).build()
forecaster.fit(train_series)  # fine-tunes from the published weights
```
