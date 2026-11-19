# stem_gnn

## Overview

StemGNN (Cao et al., "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting", NeurIPS
2020): every channel is a node; a latent correlation layer infers a graph from the window by self-attention, and
StemGNN blocks filter the window by Chebyshev polynomials of the graph Laplacian (graph Fourier transform), model
each filtered sequence in the frequency domain with gated linear units between a discrete Fourier transform and its
inverse, and apply a learned graph convolution kernel. Each block forecasts and passes a backcast to the next; the
summed forecasts of each target node are mapped to its whole horizon at once. Forecasting only.

Like the official code, the graph Fourier transform is approximated by Chebyshev polynomials of the normalized
Laplacian rather than computed by an eigendecomposition. Unlike the official code:

- the recurrent unit of the latent correlation layer reads each node's series along time and the attention is
  computed between nodes, as the paper describes; the official code runs it across the nodes with the window as
  features, with a hidden size equal to the number of nodes, and attends between hidden units;
- the Chebyshev polynomials start from the identity (the official code's first polynomial is a zero matrix) and the
  degree is taken from the symmetrized attention;
- every block but the last backcasts, generalizing the official two blocks of which only the first backcasts;
- the output layer maps to every distribution parameter of the horizon.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `StemGNNParameters`: blocks, spectral expansion, Chebyshev terms and attention settings. |
| [latent_correlation_layer.py](./latent_correlation_layer.py) | `StemGNNLatentCorrelationLayer`: graph inferred by self-attention, as Laplacian polynomials. |
| [block.py](./block.py) | `StemGNNBlock`: graph and discrete Fourier transforms, gated linear units, forecast and backcast. |
| [forecast_network.py](./forecast_network.py) | `StemGNNForecastNetwork`: latent graph, stacked blocks and output layer. |
