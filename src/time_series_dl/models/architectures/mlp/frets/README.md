# frets

## Overview

FreTS (Yi et al., "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting", NeurIPS 2023):
each value is embedded as a vector, mixed across channels and then along time by complex-valued MLPs in the frequency
domain, and every channel's embedded window is mapped to its horizon by an MLP. Forecasting only.

Unlike the official code, whose einsum `bijd,dd->bijd` multiplies by the diagonal of each weight matrix only, the
frequency MLPs use the full complex weight matrix described in the paper. Instance normalization is left to the
forecaster, as in the official code.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `FreTSParameters`: embedding and hidden sizes, sparsity threshold and channel mixing. |
| [frequency_layer.py](./frequency_layer.py) | `FreTSFrequencyLayer`: complex MLP layer applied to the spectrum along one axis. |
| [forecast_network.py](./forecast_network.py) | `FreTSForecastNetwork`: embedding, channel and temporal learners, and the output MLP. |
