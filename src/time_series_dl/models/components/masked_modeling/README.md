# masked_modeling

## Overview

Building blocks of masked-modeling pretraining (TST, PatchTST, SimMTM): the random masks of hidden values and the
decoder that reconstructs values from encoder features.

## Components

| Component | Description |
| --------- | ----------- |
| [masking.py](./masking.py) | `Masking`: random choice of the hidden values of a batch. |
| [maskings/](./maskings/README.md) | Geometric-segment and patch masks. |
| [sequence_decoder.py](./sequence_decoder.py) | `SequenceDecoder`: feature sequence linearly mapped back to the values. |
