# anomaly_transformer

## Overview

Anomaly Transformer (Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy",
ICLR 2022): anomaly-attention learns a prior association (a Gaussian kernel with a learnable scale per step) and a
series association (self-attention) of every step. A minimax strategy trains the prior to approximate the series
association and the series association to enlarge their discrepancy, which stays small for anomalies because they
can only associate with their adjacent steps. Steps are scored by the association-based criterion: the softmax of
the negative discrepancy times the reconstruction error.

Differences from the paper (following the official code):

- The two minimax phases are back-propagated before one optimizer step, as in the official code, written as one loss
  that is their sum.
- In the score, the discrepancy is summed over layers and multiplied by a temperature of 50 before the softmax
  instead of averaged over layers.
- The scale sigma is transformed by 3^(sigmoid(5 sigma) + 1e-5) - 1, and the KL divergences offset probabilities by
  1e-4, as in the official code.
- The detector averages the score of a step over all windows covering it and thresholds at a quantile of the
  training scores, instead of scoring non-overlapping windows and thresholding at a ratio of the combined training
  and test scores.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `AnomalyTransformerParameters`: Transformer sizes, the discrepancy weight and the score temperature. |
| [anomaly_attention.py](./anomaly_attention.py) | `AnomalyAttention`: series and Gaussian prior associations and the attended tokens. |
| [encoder_layer.py](./encoder_layer.py) | `AnomalyTransformerEncoderLayer`: anomaly-attention and feed-forward with residual connections. |
| [associations.py](./associations.py) | `Associations`: reconstruction and per-layer associations of a batch. |
| [anomaly_scoring_network.py](./anomaly_scoring_network.py) | `AnomalyTransformerAnomalyScoringNetwork`: minimax loss and association-based criterion. |
