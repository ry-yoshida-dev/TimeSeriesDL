# mamba

## Overview

Mamba (Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", COLM 2024): residual blocks
whose state space model has a step size and input and output matrices that depend on the input, inside an expanded,
convolved and gated branch. The selective scan is a parallel scan in PyTorch instead of the fused CUDA kernel, so
memory grows with length times inner size times state size; each block's output also goes through dropout.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MambaParameters`: sizes, expansion, convolution, layers, dropout, direction and head. |
