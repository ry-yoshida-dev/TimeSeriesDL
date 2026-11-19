# lite

## Overview

LITE (Ismail-Fawaz et al., "LITE: Light Inception with boosTing tEchniques for Time Series Classification",
DSAA 2023): a light InceptionTime whose first module combines multiplexing convolutions with fixed hand-crafted
filters, followed by two dilated depthwise separable convolutions, as a sequence encoder with the generic heads of
every task.

Differences from the paper:

- LITETime averages five LITE networks trained from different initializations; this is a single network. Train
  several and average their outputs for LITETime.
- The fixed filters are buffers; like the official code they read the sum of the channels.
- Forecasting and reconstruction use the generic heads, since every layer keeps the length.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `LITEParameters`: filters, kernel length, boosting techniques and head. |
| [custom_filters.py](./custom_filters.py) | `LITECustomFilters`: fixed increasing, decreasing and peak detection filters. |
| [inception_module.py](./inception_module.py) | `LITEInceptionModule`: multiplexing convolutions and custom filters. |
| [separable_convolution_block.py](./separable_convolution_block.py) | `LITESeparableConvolutionBlock`: dilated depthwise separable convolution. |
| [sequence_encoder.py](./sequence_encoder.py) | `LITESequenceEncoder`: the module followed by two separable blocks. |
