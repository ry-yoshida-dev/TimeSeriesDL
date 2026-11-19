# dcdetector

## Overview

DCdetector (Yang et al., "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly
Detection", KDD 2023): every channel of a window is cut into patches at several scales, and shared dual attention
layers attend over the patches (patch-wise view) and over the positions within a patch (in-patch view). A purely
contrastive loss, without reconstruction or negative pairs, pulls the in-patch view towards the patch-wise view and
pushes the patch-wise view away (each through its own stop-gradient), and the anomaly score of a step is derived from
the discrepancy of the two views' step associations.

Differences from the paper and the official code:

- The anomaly score follows the official code: the softmax over the steps of a window of the negative discrepancy
  times a temperature of 50. The paper writes the score as the discrepancy itself, which contradicts its training,
  where the patch-wise view of normal data is pushed away from the in-patch view.
- In-patch tokens gather the values at the same position of every patch, as the paper describes; the official code
  reshapes each channel into contiguous chunks instead.
- Both associations are normalized to rows that sum to one (the official code only normalizes the in-patch one), and
  the unused value projections and whole-window embedding of the official code are left out.
- The contrastive loss is zero by value; the loss the trainer monitors subtracts the detached discrepancy (the
  objective of the patch-wise view) so that best-epoch selection has a signal, while its gradient stays the
  paper's.
- Instance normalization has no affine parameters, as in effect in the official code, which creates a new RevIN
  layer in every forward pass.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `DCdetectorParameters`: patch sizes, Transformer sizes, dropout and score temperature. |
| [dual_attention.py](./dual_attention.py) | `DualAttention`: patch-wise and in-patch attention with shared projections, expanded to step associations. |
| [anomaly_scoring_network.py](./anomaly_scoring_network.py) | `DCdetectorAnomalyScoringNetwork`: multi-scale patching, contrastive loss and discrepancy score. |
