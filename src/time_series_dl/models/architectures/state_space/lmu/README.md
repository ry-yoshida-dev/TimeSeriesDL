# lmu

## Overview

Legendre Memory Unit (Voelker et al., "Legendre Memory Units: Continuous-Time Representation in Recurrent Neural
Networks", NeurIPS 2019): a scalar encoding of the input, hidden state and memory drives a Legendre memory of a
sliding window, discretized by zero-order hold, and a tanh layer of the input, hidden state and memory gives the
hidden state. The model has one layer, all its weights are learned with the default PyTorch initialization, and the
window is set relative to the input length.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `LMUParameters`: hidden size, memory order, window, discretization and head. |
| [sequence_encoder.py](./sequence_encoder.py) | `LMUSequenceEncoder`: hidden states of the cell. |
