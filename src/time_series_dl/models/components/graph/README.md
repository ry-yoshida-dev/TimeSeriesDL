# graph

## Overview

Building blocks of the graph neural networks, whose nodes are the channels of a series: graph learning layers that
infer an adjacency from node embeddings, and graph convolutions that propagate node features over a dense adjacency.

## Components

| Component | Description |
| --------- | ----------- |
| [adaptive_adjacency.py](./adaptive_adjacency.py) | `AdaptiveAdjacency`: row-normalized adjacency from one or two node embeddings. |
| [directed_graph_learner.py](./directed_graph_learner.py) | `DirectedGraphLearner`: sparse uni-directed adjacency of MTGNN. |
| [mix_hop_propagation.py](./mix_hop_propagation.py) | `MixHopPropagation`: mix-hop propagation with input retention and hop selection. |
| [diffusion_convolution.py](./diffusion_convolution.py) | `DiffusionConvolution`: powers of several transition matrices mixed pointwise. |
| [chebyshev_polynomials.py](./chebyshev_polynomials.py) | `ChebyshevPolynomials`: Chebyshev polynomial basis of a matrix. |
| [node_adaptive_graph_convolution.py](./node_adaptive_graph_convolution.py) | `NodeAdaptiveGraphConvolution`: graph convolution with node-specific weights from embeddings. |
