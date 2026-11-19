# graph_wavenet

## Overview

Graph WaveNet (Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling", IJCAI 2019): every channel is a
node; a self-adaptive adjacency `softmax(relu(E1 E2^T))` is learned from two node embeddings, and blocks of gated
dilated convolutions along time and diffusion graph convolutions feed skip connections into a pointwise output
module, which forecasts the whole horizon at once. Forecasting only.

Unlike the official code, the input has one feature per node (no time-of-day feature), and the output module maps
to every distribution parameter of the horizon. Only the adaptive adjacency is used (the official `--aptonly`
setting), since the configuration has no place for a fixed adjacency, and the unused residual convolutions of the
graph-convolution path are left out. Windows longer than the receptive field (13 steps by default) are read only
through the skip connections of their last step, as the official code assumes windows of exactly that length.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `GraphWaveNetParameters`: channels, dilation blocks, embeddings and diffusion order. |
| [layer.py](./layer.py) | `GraphWaveNetLayer`: gated temporal convolution, skip, diffusion convolution and batch normalization. |
| [forecast_network.py](./forecast_network.py) | `GraphWaveNetForecastNetwork`: adaptive adjacency, stacked layers and output module. |
