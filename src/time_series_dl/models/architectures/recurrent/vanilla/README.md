# vanilla

## Overview

Stacked LSTM, GRU or Elman layers as a sequence encoder, with the generic heads of every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `RNNParameters`: cell, hidden size, layers, direction and head. |
| [recurrent_cell.py](./recurrent_cell.py) | `RecurrentCell`: LSTM, GRU or Elman. |
| [sequence_encoder.py](./sequence_encoder.py) | `RNNSequenceEncoder`: hidden states of every step. |
