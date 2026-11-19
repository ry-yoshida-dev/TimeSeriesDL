# tcn

## Overview

Temporal convolutional network (Bai et al., 2018): dilated causal residual convolutions as a sequence encoder, with the
generic heads of every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TCNParameters`: block channels, kernel size, dropout and head. |
| [residual_block.py](./residual_block.py) | `TemporalResidualBlock`: two weight-normalized dilated causal convolutions. |
| [sequence_encoder.py](./sequence_encoder.py) | `TCNSequenceEncoder`: blocks with doubling dilation. |
