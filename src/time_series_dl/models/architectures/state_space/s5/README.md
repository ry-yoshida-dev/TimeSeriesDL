# s5

## Overview

S5 (Smith et al., "Simplified State Space Layers for Sequence Modeling", ICLR 2023): each layer runs one
multi-input multi-output state space model over all channels, with a complex diagonal state matrix initialized from
blocks of the normal part of HiPPO-LegS, a step size per state, and a parallel scan. The output stage is the GELU and
gated linear unit of S4 instead of the paper's variants.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `S5Parameters`: sizes, blocks, discretization, step sizes, layers, dropout, direction and head. |
