# conv_tran

## Overview

ConvTran (Foumani et al., "Improving position encoding of transformers for multivariate time series
classification", Data Mining and Knowledge Discovery 2024): a temporal convolution of each channel and a convolution
across channels embed every time step, the time absolute position encoding (tAPE) is added to the attention inputs,
and one attention block with the efficient relative position encoding (eRPE) and a feed-forward network encodes the
tokens. The generic heads give every task: summaries average the features over time before the output layer, as the
paper, while reconstructions (each step's features to its values) and forecasts (flattened features to the horizon)
are additions the paper does not cover.

As in the official code, the attention scores are scaled by the inverse square root of the model size rather than
the head size, the attention output is layer-normalized instead of projected, and the residual adds the tokens
without their position encoding. The vector relative position encoding of the official code is not provided.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ConvTranParameters`: sizes, dropout, position encodings and head. |
| [absolute_position_encoding.py](./absolute_position_encoding.py) | `AbsolutePositionEncoding`: tAPE, sinusoidal, learnable or none. |
| [relative_position_encoding.py](./relative_position_encoding.py) | `RelativePositionEncoding`: eRPE or none. |
| [position_encoding.py](./position_encoding.py) | `ConvTranPositionEncoding`: the absolute encoding added to the tokens. |
| [convolutional_embedding.py](./convolutional_embedding.py) | `ConvTranConvolutionalEmbedding`: temporal and cross-channel convolutions. |
| [attention.py](./attention.py) | `ConvTranAttention`: self-attention with the eRPE bias after the softmax. |
| [sequence_encoder.py](./sequence_encoder.py) | `ConvTranSequenceEncoder`: embedding, attention and feed-forward blocks. |
