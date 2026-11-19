# multi_rocket

## Overview

MultiRocket (Tan et al., "MultiRocket: multiple pooling operators and transformations for fast and effective time
series classification", Data Mining and Knowledge Discovery 2022): the MiniRocket kernels applied to the series and
to its first difference, each output pooled into four features: the proportion of positive values (PPV), the mean of
positive values (MPV), the mean of indices of positive values (MIPV) and the longest stretch of positive values
(LSPV).

Differences from the paper and its official code:

- Biases are fitted to the first batch, and only a standardized linear layer is trained by gradient descent, as in
  [MiniRocket](../mini_rocket/README.md).
- MPV is the mean of the positive differences between the output and the bias, and LSPV the longest run of positive
  differences, following their definitions in the paper rather than the details of the official code.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MultiRocketParameters`: number of features and dilations per kernel. |
| [kernel_transform.py](./kernel_transform.py) | `MultiRocketKernelTransform`: MiniRocket transforms of the series and its first difference. |
