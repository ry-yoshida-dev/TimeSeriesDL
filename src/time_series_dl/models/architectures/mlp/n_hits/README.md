# n_hits

## Overview

N-HiTS (Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting", AAAI 2023): N-BEATS
stacks whose blocks read the window pooled at different rates and interpolate coarse forecasts to the horizon. Each
target channel is forecast from its own history, with the doubly residual stacking of [n_beats](../n_beats/README.md).

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `NHITSParameters`: stacks, pooling and interpolation. |
| [stack.py](./stack.py) | `NHITSStack`: pooling kernel, downsampling factor, blocks, layers and width of one stack. |
| [pooling.py](./pooling.py) | `NHITSPooling`: max or average pooling of the input. |
| [interpolation.py](./interpolation.py) | `NHITSInterpolation`: linear or nearest interpolation of the forecast. |
| [forecast_network.py](./forecast_network.py) | `NHITSForecastNetwork`: forecasts the target channels. |
