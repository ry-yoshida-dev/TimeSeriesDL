# resnet

## Overview

Residual network (Wang et al., "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline",
IJCNN 2017): length-preserving residual convolution blocks as a sequence encoder, with the generic heads of every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ResNetParameters`: block filters, kernel lengths and head. |
| [residual_block.py](./residual_block.py) | `ResNetResidualBlock`: batch-normalized convolutions with a shortcut. |
| [sequence_encoder.py](./sequence_encoder.py) | `ResNetSequenceEncoder`: the stack of residual blocks. |
