# random_convolution

## Overview

Shared parts of the ROCKET family: fixed convolution kernels, never trained, whose outputs are pooled into features
for a trained linear layer.

## Components

| Component | Description |
| --------- | ----------- |
| [kernel_transform.py](./kernel_transform.py) | `KernelTransform`: abstract transform of sequences to features by fixed kernels. |
| [kernel_pooling.py](./kernel_pooling.py) | `KernelPooling`: PPV, max, MPV, MIPV and LSPV of a kernel's output over time. |
| [summary_network.py](./summary_network.py) | `KernelTransformSummaryNetwork`: transform, standardization and a linear layer. |
