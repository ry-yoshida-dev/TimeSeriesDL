# min_rnn

## Overview

minGRU and minLSTM (Feng et al., "Were RNNs All We Needed?", 2024): recurrent layers whose gates read only the
current input, so the hidden states follow a linear recurrence that is trained with a parallel scan in log space.
The layers are stacked in pre-normalized residual blocks with feedforward networks, and the generic heads give
forecasting, classification / regression and imputation / anomaly detection.

Differences from the paper: the initial state is zero rather than g(h_0), and each layer adds a linear output map
from its hidden size back to the model size of the residual stream. The paper gives no time series configuration,
so the block layout and defaults are a small pre-normalized residual stack.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MinRNNParameters`: cell, sizes, layers, dropout and head. |
| [minimal_recurrent_cell.py](./minimal_recurrent_cell.py) | `MinimalRecurrentCell`: minGRU or minLSTM. |
| [minimal_recurrent_layer.py](./minimal_recurrent_layer.py) | `MinimalRecurrentLayer`: one layer, in parallel or sequential form. |
| [log_space_scan.py](./log_space_scan.py) | `LogSpaceScan`: parallel scan of a positive linear recurrence in log space. |
| [sequence_encoder.py](./sequence_encoder.py) | `MinRNNSequenceEncoder`: embedding and residual blocks. |
