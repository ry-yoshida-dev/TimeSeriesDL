# tst

## Overview

TST (Zerveas et al., "A Transformer-based Framework for Multivariate Time Series Representation Learning",
KDD 2021): each time step's values are projected linearly and scaled by the square root of the model size, a
learnable position encoding is added, and post-norm Transformer encoder layers with batch normalization (or layer
normalization) are followed by a final activation. The generic heads give every task: summaries flatten every
step's features after dropout, as the official classification and regression output layer, reconstructions map
each step's features to its values, as the official imputation output layer, and forecasts map the flattened
features linearly to the horizon, which the paper does not cover.

Unlike the paper, windows have a fixed length, so there are no padding masks; the masked-value self-supervised
pretraining is not provided (the reconstruction task trains imputation directly), and only the learnable position
encoding is offered.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TSTParameters`: sizes, layers, dropout, activation, normalization and head. |
| [encoder_layer.py](./encoder_layer.py) | `TSTEncoderLayer`: post-norm self-attention and feed-forward network. |
| [sequence_encoder.py](./sequence_encoder.py) | `TSTSequenceEncoder`: scaled input projection, learnable positions and encoder layers. |
