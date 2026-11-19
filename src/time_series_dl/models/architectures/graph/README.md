# graph

## Overview

Graph neural networks whose nodes are the channels of a series. Since no adjacency is given with the data, every
model infers its graph: from learned node embeddings, from the window itself, or as the fully connected graph of all
values. Every channel is a node, also when it is not a forecast target.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation | Anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :--------: | :---------------: |
| [Graph WaveNet (Wu et al., 2019)](./graph_wavenet/README.md) | `GraphWaveNetParameters` | ✓ | | | |
| [MTGNN (Wu et al., 2020)](./mtgnn/README.md) | `MTGNNParameters` | ✓ | ✓ | ✓ | ✓ |
| [AGCRN (Bai et al., 2020)](./agcrn/README.md) | `AGCRNParameters` | ✓ | | | |
| [StemGNN (Cao et al., 2020)](./stem_gnn/README.md) | `StemGNNParameters` | ✓ | | | |
| [FourierGNN (Yi et al., 2023)](./fourier_gnn/README.md) | `FourierGNNParameters` | ✓ | | | |
