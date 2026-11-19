# layers

## Overview

Building blocks and options shared by several models.

## Components

| Component | Description |
| --------- | ----------- |
| [activation.py](./activation.py) | `Activation`: ReLU, GELU, SiLU or tanh. |
| [temporal_pooling.py](./temporal_pooling.py) | `TemporalPooling`: mean, max, last step or flattening of a feature sequence. |
| [multilayer_perceptron.py](./multilayer_perceptron.py) | `MultilayerPerceptron`: fully connected layers over the last dimension. |
| [series_decomposition.py](./series_decomposition.py) | `SeriesDecomposition`: moving-average trend and seasonal remainder. |
| [positional_encoding.py](./positional_encoding.py) | `SinusoidalPositionalEncoding`: fixed position encoding of the Transformer. |
| [temporal_linear.py](./temporal_linear.py) | `TemporalLinear`: linear map along time, shared or individual per channel. |
| [same_length_convolution.py](./same_length_convolution.py) | `SameLengthConvolution`: 1D convolution that keeps the length, also for even kernels. |
| [convolution_block.py](./convolution_block.py) | `ConvolutionBlock`: length-keeping convolution, batch normalization and optional ReLU. |
