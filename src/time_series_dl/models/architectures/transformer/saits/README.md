# saits

## Overview

SAITS (Du et al., "SAITS: Self-attention-based imputation for time series", Expert Systems with Applications, 2023):
two diagonally masked self-attention (DMSA) blocks read the values together with the observed mask; the second block
refines the first block's imputation, and the two estimates are combined by weights learned from the mask and the
second block's attention map. Training joins the observed reconstruction task (ORT, the mean absolute error of the
three estimates on the observed inputs, computed by the network) and the masked imputation task (MIT, the imputer's
loss on the observed values it hid). Imputation only.

Differences from the paper and PyPOTS:

- The MIT loss is the imputer's `loss` and its artificial masking rate is the imputer's `mask_ratio`; the paper uses
  `MAELoss` and a rate of 0.2.
- Each layer has its own weights; the parameter sharing between layer groups of the original code is not offered.
- Layers normalize after each residual connection, as in PyPOTS's encoder layer.
- The defaults are those of the PyPOTS usage example; the paper tunes them per dataset.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `SAITSParameters`: layer sizes, dropouts and the ORT and MIT weights. |
| [diagonally_masked_attention.py](./diagonally_masked_attention.py) | `SAITSDiagonallyMaskedAttention`: multi-head self-attention in which no step attends to itself. |
| [encoder_layer.py](./encoder_layer.py) | `SAITSEncoderLayer`: diagonally masked attention and a feed-forward network. |
| [block.py](./block.py) | `SAITSBlock`: joint embedding of values and mask, position encoding and encoder layers. |
| [masked_reconstruction_network.py](./masked_reconstruction_network.py) | `SAITSMaskedReconstructionNetwork`: the two blocks, their weighted combination and the ORT loss. |
