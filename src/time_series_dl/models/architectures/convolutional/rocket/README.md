# rocket

## Overview

ROCKET (Dempster et al., "ROCKET: Exceptionally fast and accurate time series classification using random
convolutional kernels", Data Mining and Knowledge Discovery 2020): many random kernels of random length, dilation,
padding, bias and channels; each output is pooled into its proportion of positive values (PPV) and its maximum.

Differences from the paper and its sktime implementation:

- The kernels are fixed buffers and only a linear layer on the features is trained, by gradient descent with the
  task's loss, instead of a ridge regression; for classification this is the logistic regression variant of the
  paper. Features are standardized by batch normalization without affine parameters, the counterpart of the
  normalization in front of the ridge and logistic regressions.
- Each series is standardized per channel inside the transform, as sktime does by default.
- Kernels are stored centered in the longest length and grouped by dilation, so they run as one convolution per
  dilation. When a series is too short for a kernel without padding to fit, that kernel pools every step.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `RocketParameters`: number of kernels, kernel lengths and normalization. |
| [kernel_transform.py](./kernel_transform.py) | `RocketKernelTransform`: the random kernels and their PPV and max features. |
