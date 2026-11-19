# fits

## Overview

FITS (Xu et al., "FITS: Modeling Time Series with 10k Parameters", ICLR 2024): the window's spectrum is cut by a
low-pass filter, a complex-valued linear layer interpolates it to the spectrum of the window extended by the horizon,
and the inverse transform, scaled by the length ratio, gives the extended series, whose last steps are the forecast.
Reconstruction interpolates the spectrum of the window onto itself.

Unlike the official code, the reversible instance normalization is left to the forecasting task, the inverse
transform is taken at the exact extended length (the official code drops one step when it is odd), each distribution
parameter has its own complex output frequencies, and the loss is on the horizon only (the official code can also
supervise the reconstructed window). The cutoff frequency follows the official formula
`(input_length // base_period + 1) * harmonic_order + 10`, capped at the number of frequencies of the window.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `FITSParameters`: base period, harmonic order and per-channel weights. |
| [frequency_interpolation.py](./frequency_interpolation.py) | `FITSFrequencyInterpolation`: low-pass filter, complex linear map and inverse transform. |
| [forecast_network.py](./forecast_network.py) | `FITSForecastNetwork`: extends each target channel and keeps the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `FITSReconstructionNetwork`: filters and maps each channel onto itself. |
