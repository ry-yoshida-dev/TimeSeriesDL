# lstnet

## Overview

LSTNet (Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks", SIGIR 2018): a
convolution over time and channels, a GRU, and a recurrent-skip GRU that links steps one period apart, plus a
highway autoregressive component on each target channel. Forecasting and classification / regression.

Differences from the paper and the official code: the forecast is direct multi-step, mapping the final features
(and, for the highway, the last values of each target) to the whole horizon and every distribution parameter
instead of one future step; the official code's output sigmoid is dropped, since the task normalizes each window;
the skip component reads every whole period of the convolved window; the attention variant (LSTNet-Attn) is not
implemented. Summary networks map the features linearly to the output vector, without the highway.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `LSTNetParameters`: convolution, recurrent sizes, skip period, highway window and dropout. |
| [sequence_encoder.py](./sequence_encoder.py) | `LSTNetSequenceEncoder`: convolution, GRU and recurrent-skip GRU into one feature vector. |
| [forecast_network.py](./forecast_network.py) | `LSTNetForecastNetwork`: features and highway mapped to the horizon. |
