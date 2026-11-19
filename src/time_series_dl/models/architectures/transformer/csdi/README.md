# csdi

## Overview

CSDI (Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation",
NeurIPS 2021): a conditional diffusion model that imputes the unobserved entries of a (features, time) array from the
observed ones, with a Transformer over time and one over features in every residual layer. Forecasting imputes the
horizon of the target channels from the past window. It is a generative model: it is trained with the denoising loss
and forecasts with `SampleObjective`, and reads no covariates. `CSDIConditionalImputer` is the mask-aware core that
an imputation task can train with random conditional masks and sample with the observed mask.

Differences from the paper and the authors' code: the task normalizes each window instead of normalizing the dataset;
every feature is modeled (the forecasting code samples a subset of 64 features while training); time positions are
those of the window; the attention along time also runs on length-one arrays, and the attention along features is
left out for a single feature, where it is the identity; and the validation loss is that of one random diffusion step
instead of the average over all steps.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `CSDIParameters`: channels, layers, heads, embedding sizes and noise schedule; builds the forecast network and the imputer. |
| [generative_forecast_network.py](./generative_forecast_network.py) | `CSDIForecastNetwork`: forecasting as imputation of the horizon. |
| [conditional_imputer.py](./conditional_imputer.py) | `CSDIConditionalImputer`: denoising loss of the target entries and sampling of the unconditioned entries for any masks. |
| [denoiser.py](./denoiser.py) | `CSDIDenoiser`: predicts the noise from the conditioned values, the noisy values, the mask and the diffusion step. |
| [residual_layer.py](./residual_layer.py) | `CSDIResidualLayer`: temporal and feature Transformers and the gated output. |
| [side_information.py](./side_information.py) | `CSDISideInformation`: time and feature embeddings and the conditional mask. |

## Examples

```python
imputer = CSDIParameters().build_conditional_imputer(num_features=channels)
loss = imputer.compute_loss(values, observed_mask, conditional_mask)  # (batch, features, length) each
samples = imputer.impute(values, observed_mask, num_samples=100)  # (batch, 100, features, length)
```
