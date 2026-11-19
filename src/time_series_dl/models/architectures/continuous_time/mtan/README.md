# mtan

## Overview

mTAN (Shukla and Marlin, "Multi-Time Attention Networks for Irregularly Sampled Time Series", ICLR 2021). Multi-time
attention re-represents the observed values and masks at fixed reference times through learned time embeddings,
with a separate softmax per channel over the times where it is observed. Classification and regression use the
mTAND-Enc encoder (attention, then a GRU whose last state feeds the output layers); interpolation and forecasting use
the variational encoder-decoder mTAND-Full.

Differences from the paper and the authors' code: the time embeddings are always learned (the fixed sinusoidal
option is left out); reference times span [0, 1], the range of the scaled training times; mTAND-Full is trained with
a single-sample evidence lower bound (no importance-weighted samples, `k_iwae` = 1) at a constant `kl_weight`
instead of the KL annealing schedule, and predicts by decoding the posterior means; the classifier is mTAND-Enc
alone rather than the jointly trained mTAND-Full classifier. Shared defaults follow the authors' PhysioNet
classification script, the latent size and output standard deviation their interpolation script.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `MTANParameters`: sizes, heads, reference points, latent state, likelihood and head. |
| [time_embedding.py](./time_embedding.py) | `LearnedTimeEmbedding`: linear and learned sinusoidal components of a time. |
| [multi_time_attention.py](./multi_time_attention.py) | `MultiTimeAttention`: attention between time embeddings with a per-channel masked softmax. |
| [irregular_sequence_encoder.py](./irregular_sequence_encoder.py) | `MTANSequenceEncoder`: mTAND-Enc. |
| [irregular_interpolation_network.py](./irregular_interpolation_network.py) | `MTANInterpolationNetwork`: mTAND-Full and its evidence lower bound. |
