# vanilla

## Overview

Transformer encoder (Vaswani et al., "Attention Is All You Need", NeurIPS 2017): value embedding with sinusoidal
position encoding and self-attention layers in post- or pre-norm order as a sequence encoder, with the generic heads
of every task.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TransformerParameters`: sizes, layers, normalization order and head. |
| [sequence_encoder.py](./sequence_encoder.py) | `TransformerSequenceEncoder`: embedded tokens through the encoder layers. |
