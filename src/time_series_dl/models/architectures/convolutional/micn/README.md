# micn

## Overview

MICN (Wang et al., "MICN: Multi-scale Isometric Convolution Network for Long-term Time Series Forecasting",
ICLR 2023): the series is split into a trend, forecast by a linear map along time, and a seasonal part, modeled by
multi-scale isometric convolutions that combine local features (strided convolutions) with global correlations
(an isometric convolution spanning the whole downsampled sequence).

The layers follow the Time-Series-Library implementation. Differences from the paper and that implementation:

- The multi-scale decomposition averages the moving averages, as in the Time-Series-Library, instead of the
  learned weighting of the original code.
- No calendar covariates are embedded; the token embedding adds the sinusoidal position encoding only.
- One dropout probability is used for the embedding, the MIC layers (fixed at 0.05 in the Time-Series-Library) and
  the summary head.
- Forecasts of several distribution parameters add the trend forecast only to the parameters in the scale of the
  values, and only target channels are projected; the other channels are inputs only.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MICNParameters`: token size, layers, scales and dropout. |
| [multiscale_decomposition.py](./multiscale_decomposition.py) | `MICNMultiScaleDecomposition`: mean of moving-average trends of several lengths. |
| [isometric_layer.py](./isometric_layer.py) | `MICNIsometricLayer`: local downsampling, isometric and upsampling convolutions per scale, merged. |
| [seasonal_prediction.py](./seasonal_prediction.py) | `MICNSeasonalPrediction`: embedding, MIC layers and a projection of each step. |
| [forecast_network.py](./forecast_network.py) | `MICNForecastNetwork`: linear trend forecast plus the seasonal prediction over the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `MICNReconstructionNetwork`: seasonal prediction of the sequence plus its trend. |
| [summary_network.py](./summary_network.py) | `MICNSummaryNetwork`: the reconstruction-like output flattened to the output vector. |
