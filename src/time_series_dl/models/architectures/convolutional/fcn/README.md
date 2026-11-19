# fcn

## Overview

Fully convolutional network (Wang et al., "Time Series Classification from Scratch with Deep Neural Networks: A Strong
Baseline", IJCNN 2017): length-preserving convolution blocks with batch normalization as a sequence encoder, with the
generic heads of every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `FCNParameters`: block filters, kernel lengths and head. |
| [sequence_encoder.py](./sequence_encoder.py) | `FCNSequenceEncoder`: the stack of convolution blocks. |
