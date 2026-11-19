# mtgnn

## Overview

MTGNN (Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks", KDD 2020):
every channel is a node of a graph that a graph learning layer infers from node embeddings; layers of gated dilated
inception convolutions along time and mix-hop propagation along and against the learned edges feed skip connections
into a pointwise output module, which forecasts the whole horizon at once. The node features also feed a
reconstruction and a summary head.

Unlike the official code, the input has one feature per node (no time-of-day feature), the output module maps to
every distribution parameter of the horizon, the unused residual convolutions of the graph-convolution path are
left out, and the tie-breaking noise of the top-k selection is added only while training. No fixed adjacency can be
given, since the configuration has no place for one, and curriculum learning and node subsampling, which belong to
training, are not implemented. The reconstruction and summary heads are additions of this package.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MTGNNParameters`: layer sizes, graph learning and propagation settings. |
| [dilated_inception.py](./dilated_inception.py) | `MTGNNDilatedInception`: dilated convolutions with four kernel sizes, concatenated. |
| [layer.py](./layer.py) | `MTGNNLayer`: gated temporal convolution, skip, mix-hop propagation and layer normalization. |
| [encoder.py](./encoder.py) | `MTGNNEncoder`: learned graph, stacked layers and output module up to node features. |
| [forecast_network.py](./forecast_network.py) | `MTGNNForecastNetwork`: each target node's features to its horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `MTGNNReconstructionNetwork`: each node's features to its sequence. |
| [summary_network.py](./summary_network.py) | `MTGNNSummaryNetwork`: all flattened node features to the output vector. |
