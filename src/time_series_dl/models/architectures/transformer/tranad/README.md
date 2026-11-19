# tranad

## Overview

TranAD (Tuli et al., "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data",
VLDB 2022): a Transformer encoder reads the window together with a focus score, and two decoders reconstruct it. In
the second phase the first decoder's reconstruction error becomes the focus score (self-conditioning), and the two
decoders play an adversarial game on the second-phase reconstruction. Steps are scored by the mean of the first- and
second-phase reconstruction errors.

Differences from the paper and the official code:

- The paper's phase weight epsilon^-n is replaced by 1/n, as in the official code; the official code drops the
  adversarial sign, which is kept here.
- The two phase losses are combined into one loss whose gradient equals the paper's for each sub-network (see
  `TranADAnomalyScoringNetwork`), so one optimizer trains them.
- There is no separate encoder of the complete sequence up to the window: the window is the context. The decoders
  reconstruct every step of the window instead of only its last step, so every step is scored by every window that
  covers it.
- Each decoder has its own output layer (the official code shares one), and the position encoding is the standard
  sinusoidal one.
- The loss value the trainer monitors is the unweighted first-phase reconstruction error; its gradient is the
  paper's.

The decoders end with a sigmoid as in the paper, so scale the channels to [0, 1] as the paper does, with
`scaling=MinMaxScalingParameters()` in `AnomalyDetectorSettings`.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TranADParameters`: feed-forward size, number of layers and dropout. |
| [encoder.py](./encoder.py) | `TranADEncoder`: window and focus score to the memory. |
| [encoder_layer.py](./encoder_layer.py) | `TranADEncoderLayer`: self-attention and feed-forward without normalization. |
| [decoder.py](./decoder.py) | `TranADDecoder`: memory to the reconstructed window. |
| [decoder_layer.py](./decoder_layer.py) | `TranADDecoderLayer`: self-attention, memory attention and feed-forward without normalization. |
| [anomaly_scoring_network.py](./anomaly_scoring_network.py) | `TranADAnomalyScoringNetwork`: two-phase adversarial loss and score. |
