# agcrn

## Overview

AGCRN (Bai et al., "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting", NeurIPS 2020): every
channel is a node; a data adaptive graph `softmax(relu(E E^T))` is generated from one node embedding, and stacked
gated recurrent units whose linear maps are graph convolutions with node adaptive weights `E W_pool` read the
window. The last hidden state of each target node is mapped pointwise to its whole horizon at once. Forecasting
only.

Unlike the official code, the output layer maps to every distribution parameter of the horizon, and the node
embeddings keep their standard normal initialization while the weight pools use Xavier initialization (the official
training script re-initializes every matrix with Xavier). No fixed adjacency can be given, matching the paper, which
needs none.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `AGCRNParameters`: hidden size, layers, embedding size and Chebyshev terms. |
| [cell.py](./cell.py) | `AGCRNCell`: gated recurrent unit with node adaptive graph convolutions. |
| [forecast_network.py](./forecast_network.py) | `AGCRNForecastNetwork`: adaptive graph, recurrent layers and pointwise output. |
