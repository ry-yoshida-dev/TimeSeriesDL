# s4

## Overview

S4 (Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022): a deep state space
model whose layers give each channel a diagonal plus low-rank state matrix initialized from HiPPO-LegS, applied as a
long convolution. As in The Annotated S4, the complex state of size `state_size` has no conjugate pairs and the
kernel is its real part, and every channel learns its own state matrix. The parameters of the state space models
use the optimizer's learning rate and weight decay, which the paper lowers and removes for them.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `S4Parameters`: sizes, initial step sizes, layers, dropout, direction and head. |
