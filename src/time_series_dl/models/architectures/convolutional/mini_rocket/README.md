# mini_rocket

## Overview

MiniRocket (Dempster et al., "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
Classification", KDD 2021): 84 fixed kernels of length 9 at exponentially spread dilations, each with several biases
taken from quantiles of its output, pooled into proportions of positive values (PPV).

The transform follows the official multivariate implementation. Differences:

- There is no separate fit step: the first batch the network sees fits the biases, each from the output on a random
  sequence of that batch, and they are kept fixed afterwards as buffers, so checkpoints keep them.
- Only a linear layer on the features is trained, by gradient descent with the task's loss, instead of a ridge
  regression; features are standardized by batch normalization without affine parameters.
- The 84 kernels of each dilation run as one convolution; series shorter than 9 steps use dilation 1, and kernels
  without padding pool every step when they do not fit the series.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MiniRocketParameters`: number of features and dilations per kernel. |
| [kernel_transform.py](./kernel_transform.py) | `MiniRocketKernelTransform`: kernels, dilations, fitted biases and pooling. |
