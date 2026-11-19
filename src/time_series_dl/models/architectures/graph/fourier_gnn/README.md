# fourier_gnn

## Overview

FourierGNN (Yi et al., "FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective",
NeurIPS 2023): every (channel, step) value of the window is a node of one fully connected hypervariate graph and is
embedded by a learned vector. Stacked Fourier graph operators, complex matrix multiplications of the discrete
Fourier transform taken over all nodes, perform graph convolutions of increasing order, whose soft-shrunk outputs
are summed with the input. After the inverse transform, each target channel's steps are projected to a few features
and a feed-forward network maps them to its whole horizon at once. Forecasting only.

Unlike the official code, each operator multiplies by a full complex weight matrix as the paper defines it (the
official `einsum('bli,ii->bli')` uses only the diagonal of each weight matrix), the hidden size factor, which the
official code supports only at 1, is left out, the number of operators is configurable (three in the official
code), and the output network maps to every distribution parameter of the horizon. The non-target channels are
nodes of the graph but are not forecast.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `FourierGNNParameters`: embedding size, operators, sparsity threshold and output sizes. |
| [fourier_graph_operators.py](./fourier_graph_operators.py) | `FourierGNNGraphOperators`: stacked Fourier graph operators with soft shrinkage. |
| [forecast_network.py](./forecast_network.py) | `FourierGNNForecastNetwork`: value embedding, operators in the Fourier domain and output network. |
