# state_space_layers

## Overview

Implementations of `StateSpaceLayer`, each including its own output projection.

## Components

| Component | Description |
| --------- | ----------- |
| [convolutional.py](./convolutional.py) | `ConvolutionalStateSpaceLayer`: an `SSMKernel` applied as a long convolution, for S4 and S4D. |
| [s5.py](./s5.py) | `S5StateSpaceLayer`: one multi-input multi-output diagonal model over all channels. |
| [lru.py](./lru.py) | `LRUStateSpaceLayer`: Linear Recurrent Unit, a diagonal recurrence parametrized in discrete time. |
| [mamba.py](./mamba.py) | `MambaStateSpaceLayer`: Mamba block with a selective state space model. |
| [mamba2.py](./mamba2.py) | `Mamba2StateSpaceLayer`: Mamba-2 block with the chunked state space duality algorithm. |
