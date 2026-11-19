# film

## Overview

FiLM (Zhou et al., "FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting",
NeurIPS 2022): each target channel is projected onto Legendre polynomials by a fixed HiPPO-LegT memory over windows
of several multiples of the horizon, the memory states are filtered by a frequency enhanced layer, the last state is
reconstructed as the horizon, and a linear layer mixes the scales. Forecasting only.

Unlike Time-Series-Library, the reversible instance normalization is left to the forecaster, the forecast comes from
the last memory state, the Legendre polynomials are evaluated at the centre of each step, and the frequency weights
have no low-rank approximation.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `FiLMParameters`: memory size, scales and Fourier modes. |
| [legendre_projection.py](./legendre_projection.py) | `FiLMLegendreProjection`: LegT memory states and the reconstruction of the window. |
| [frequency_enhanced_layer.py](./frequency_enhanced_layer.py) | `FiLMFrequencyEnhancedLayer`: complex linear map of the lowest Fourier modes. |
| [scale_branch.py](./scale_branch.py) | `FiLMScaleBranch`: projection, filtering and reconstruction at one scale. |
| [forecast_network.py](./forecast_network.py) | `FiLMForecastNetwork`: target channels through every scale, mixed into the parameters. |
