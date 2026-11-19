# masked_reconstruction

## Overview

Masked-value reconstruction pretraining of TST (Zerveas et al., "A Transformer-based Framework for Multivariate
Time Series Representation Learning", KDD 2021) and of the self-supervised PatchTST (Nie et al., "A Time Series is
Worth 64 Words", ICLR 2023), around the encoder of any model: hidden values are set to zero and reconstructed by a
linear map of each step's features, with the mean squared error of the hidden values as the loss. Geometric masks
of each channel follow TST, patch masks follow PatchTST.

PatchTST's own channel-independent patch encoder is not a sequence encoder, so its pretraining is approximated by
patch masks around any length-keeping encoder (an encoder that shortens the sequence is decoded from all its
flattened features); PatchTST's per-patch reconstruction head and instance normalization of each sequence are not
reproduced. Sequences have a fixed length, so TST's padding masks are not needed.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MaskedReconstructionParameters`: encoder, masking and instance pooling. |
| [pretraining_network.py](./pretraining_network.py) | `MaskedReconstructionPretrainingNetwork`: zeroed hidden values and their reconstruction loss. |
