# csdi

## Overview

CSDI (Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation",
NeurIPS 2021): a conditional diffusion model that imputes the unobserved entries of a (features, time) array from the
observed ones, with a Transformer over time and one over features in every residual layer. Forecasting imputes the
horizon of the target channels from the past window. It is a generative model: it is trained with the denoising loss
and forecasts with `SampleObjective`, and reads no covariates. For imputation, the network trains only with its
denoising loss on a random conditional mask drawn from the observed values (the paper's random strategy) and fills
each missing value with the median of `num_imputation_samples` samples.

Differences from the paper and the authors' code: the task normalizes each window instead of normalizing the dataset;
every feature is modeled (the forecasting code samples a subset of 64 features while training); time positions are
those of the window; the attention along time also runs on length-one arrays, and the attention along features is
left out for a single feature, where it is the identity; and the validation loss is that of one random diffusion step
instead of the average over all steps. Imputation uses the median of 20 samples by default instead of 100, and its
validation loss is the imputer's point loss on the values the task hides, not the denoising loss.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `CSDIParameters`: channels, layers, heads, embedding sizes and noise schedule; builds the forecast network and the imputer. |
| [generative_forecast_network.py](./generative_forecast_network.py) | `CSDIForecastNetwork`: forecasting as imputation of the horizon. |
| [masked_reconstruction_network.py](./masked_reconstruction_network.py) | `CSDIMaskedReconstructionNetwork`: imputation trained on random conditional masks and imputing by the median of samples. |
| [conditional_imputer.py](./conditional_imputer.py) | `CSDIConditionalImputer`: denoising loss of the target entries and sampling of the unconditioned entries for any masks. |
| [denoiser.py](./denoiser.py) | `CSDIDenoiser`: predicts the noise from the conditioned values, the noisy values, the mask and the diffusion step. |
| [residual_layer.py](./residual_layer.py) | `CSDIResidualLayer`: temporal and feature Transformers and the gated output. |
| [side_information.py](./side_information.py) | `CSDISideInformation`: time and feature embeddings and the conditional mask. |

## Examples

```python
imputer = ImputerSettings(model=CSDIParameters(num_imputation_samples=100)).build()
_ = imputer.fit(sequences)  # (num_sequences, length, channels); NaN marks the missing values
filled = imputer.impute(sequences)

# The mask-aware core, for (batch, features, length) arrays.
core = CSDIParameters().build_conditional_imputer(num_features=channels)
loss = core.compute_loss(values, observed_mask, conditional_mask)  # (batch, features, length) each
samples = core.impute(values, observed_mask, num_samples=100)  # (batch, 100, features, length)
```
