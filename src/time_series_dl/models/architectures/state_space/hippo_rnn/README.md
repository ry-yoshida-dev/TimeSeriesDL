# hippo_rnn

## Overview

HiPPO-RNN (Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections", NeurIPS 2020): a GRU cell that
reads the input and a HiPPO memory, and writes a learned scalar of its hidden state into the memory at every step.
The memory follows the scaled Legendre (LegS), translated Legendre (LegT), Laguerre or Fourier measure. The model
has one recurrent layer, and a translated measure's window is set relative to the input length.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `HiPPORNNParameters`: measure, hidden and memory sizes, window, discretization and head. |
| [sequence_encoder.py](./sequence_encoder.py) | `HiPPORNNSequenceEncoder`: hidden states of the cell with its memory. |
