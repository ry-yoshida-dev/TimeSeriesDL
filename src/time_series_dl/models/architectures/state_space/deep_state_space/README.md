# deep_state_space

## Overview

The encoder shared by S4, S4D, S5, LRU, Mamba and Mamba-2: a linear embedding, residual blocks
x + dropout(layer(norm(x))) around a `StateSpaceLayer`, optionally with a second layer that reads the sequence
backward, and a final normalization. The models differ in the layer and normalization their parameters build
(layer normalization, or RMS normalization for the Mamba models), and get every task from the generic heads.

## Components

| Component | Description |
| --------- | ----------- |
| [sequence_encoder.py](./sequence_encoder.py) | `DeepStateSpaceSequenceEncoder`: embedding, residual state space blocks and final normalization. |
