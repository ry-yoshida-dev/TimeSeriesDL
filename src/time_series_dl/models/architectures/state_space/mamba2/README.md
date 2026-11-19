# mamba2

## Overview

Mamba-2 (Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State
Space Duality", ICML 2024): Mamba blocks with a scalar state matrix per head, input and output matrices shared by
every head (one group), parallel input projections and a gated RMS normalization, computed by the chunked state
space duality algorithm in PyTorch.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `Mamba2Parameters`: sizes, heads, convolution, chunk length, layers, dropout, direction and head. |
