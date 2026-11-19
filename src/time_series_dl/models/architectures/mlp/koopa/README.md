# koopa

## Overview

Koopa (Liu et al., "Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors", NeurIPS 2023):
stacked blocks split their input with a Fourier filter into a time-invariant part, forecast by a learned global
Koopman operator, and a time-variant part, forecast by a Koopman operator fitted to each window's segments; each
block removes the time-variant part it reconstructs, and the block forecasts are summed. Forecasting only.

Unlike Time-Series-Library, the Non-stationary Transformer normalization is left to the forecaster; the
time-invariant frequencies, which Time-Series-Library ranks over the whole training set, are ranked by the mean
amplitude of the first training batch and then kept fixed; each window's operator is fitted with a pseudo-inverse
instead of a least-squares solver; the reconstruction of the time-variant part is aligned with the end of the padded
window; and a linear layer per distribution parameter maps the forecast to the parameters.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `KoopaParameters`: blocks, operator and hidden sizes, and the time-invariant fraction. |
| [fourier_filter.py](./fourier_filter.py) | `KoopaFourierFilter`: split into time-variant and time-invariant parts by a fixed frequency mask. |
| [dynamic_mode_decomposition.py](./dynamic_mode_decomposition.py) | `KoopaDynamicModeDecomposition`: one-step operator fitted to a sequence, its reconstruction and rollout. |
| [time_invariant_predictor.py](./time_invariant_predictor.py) | `KoopaTimeInvariantPredictor`: learned global operator over each channel's window. |
| [time_variant_predictor.py](./time_variant_predictor.py) | `KoopaTimeVariantPredictor`: per-window operator over segments of every channel. |
| [forecast_network.py](./forecast_network.py) | `KoopaForecastNetwork`: the stacked blocks and the parameter layer. |
