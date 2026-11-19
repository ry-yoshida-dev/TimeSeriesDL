# modern_tcn

## Overview

ModernTCN (Luo and Wang, "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis",
ICLR 2024): each variable is cut into patches, and blocks mix them with a large-kernel depthwise convolution along
time and two grouped pointwise feed-forward networks, ConvFFN1 over the features of each variable and ConvFFN2 over
the variables of each feature.

The layers follow the official code. Differences:

- Instance normalization (RevIN) is applied by the forecasting task, not by the network.
- The large and small depthwise kernels are kept separate (the training form); the official code can merge them
  for inference with the same outputs. The unused multi-scale head option is not implemented.
- Head dropout is applied before the output layer instead of after it, as in the other patch models here.
- Series shorter than one patch stride are extended by repeating their last value to one whole patch, and stages
  with fewer patches than the downsampling needs repeat their last patches, so every input length works.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ModernTCNParameters`: stages, patching, downsampling, feed-forward ratio and dropout. |
| [stage.py](./stage.py) | `ModernTCNStage`: number of blocks, kernel lengths and feature size of one stage. |
| [large_kernel_convolution.py](./large_kernel_convolution.py) | `ModernTCNLargeKernelConvolution`: large plus small depthwise convolution. |
| [block.py](./block.py) | `ModernTCNBlock`: DWConv, ConvFFN1 and ConvFFN2 with a residual connection. |
| [encoder.py](./encoder.py) | `ModernTCNEncoder`: patch embedding, downsampling and stages. |
| [forecast_network.py](./forecast_network.py) | `ModernTCNForecastNetwork`: each target variable's features to its horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `ModernTCNReconstructionNetwork`: each variable's features to its sequence. |
| [summary_network.py](./summary_network.py) | `ModernTCNSummaryNetwork`: all features to the output vector. |
