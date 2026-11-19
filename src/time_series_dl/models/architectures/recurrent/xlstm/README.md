# xlstm

## Overview

xLSTM (Beck et al., "xLSTM: Extended Long Short-Term Memory", NeurIPS 2024): a stack of residual blocks over the
embedded time steps, each an mLSTM block (matrix memory, pre up-projection, trained in its parallel form) or an
sLSTM block (exponential gating with normalizer and stabilizer states, block-diagonal recurrent heads, post
up-projection feedforward network). Which blocks are sLSTM blocks is configurable. The generic heads give
forecasting, classification / regression and imputation / anomaly detection.

Differences from the paper and the official code: the sLSTM cell runs step by step in PyTorch instead of the fused
CUDA kernel, and its first stabilizer is max(input gate, log forget gate) from a zero state instead of the input gate
(the outputs are the same up to rounding); the mLSTM parallel form is evaluated without chunking, so its memory grows
with the square of the window length; the linear layers of the official code's language model (token embedding and
weight tying) are replaced by a linear embedding of each step and the generic heads.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `XLSTMParameters`: model size, stack and head. |
| [stack_parameters.py](./stack_parameters.py) | `XLSTMStackParameters`: blocks, sLSTM positions, heads and sizes of a stack; shared with xLSTMTime. |
| [block_stack.py](./block_stack.py) | `XLSTMBlockStack`: the blocks in order and a final normalization. |
| [sequence_encoder.py](./sequence_encoder.py) | `XLSTMSequenceEncoder`: step embedding followed by the stack. |
| [mlstm_block.py](./mlstm_block.py) | `MLSTMBlock`: residual mLSTM block with pre up-projection. |
| [mlstm_cell.py](./mlstm_cell.py) | `MLSTMCell`: matrix memory in parallel and recurrent form. |
| [slstm_block.py](./slstm_block.py) | `SLSTMBlock`: residual sLSTM block with a gated feedforward network. |
| [slstm_cell.py](./slstm_cell.py) | `SLSTMCell`: scalar memory with exponential gating and block-diagonal recurrence. |
| [headwise_linear.py](./headwise_linear.py) | `HeadwiseLinear`: block-diagonal linear map over heads. |
| [headwise_layer_norm.py](./headwise_layer_norm.py) | `HeadwiseLayerNorm`: layer normalization of each head. |
| [gated_feedforward.py](./gated_feedforward.py) | `GatedFeedForward`: GELU-gated feedforward network. |

## Examples

```python
from time_series_dl.models.architectures.recurrent.xlstm import XLSTMParameters, XLSTMStackParameters

mlstm_only = XLSTMParameters(stack=XLSTMStackParameters(num_blocks=4, slstm_block_indices=()))
alternating = XLSTMParameters(stack=XLSTMStackParameters(num_blocks=4, slstm_block_indices=(1, 3)))
```
