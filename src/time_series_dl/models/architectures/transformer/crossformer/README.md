# crossformer

## Overview

Crossformer (Zhang & Yan, "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series
Forecasting", ICLR 2023): each channel is cut into segments embedded as tokens (dimension-segment-wise embedding),
two-stage attention mixes the segments of each channel and then the channels through routers, and a hierarchical
encoder-decoder merges adjacent segments at coarser scales and sums the predictions of every scale. Reconstructions
and summaries flatten the coarsest encoder scale, as in the Time-Series-Library.

Differences from the paper and the official code: the forecast predicts only the target channels, each output
segment carries every distribution parameter, and the forecast channels are decoded from all input channels. Unlike
the Time-Series-Library, the segment embedding follows the official code (front padding with the first value, a
linear map with bias and no extra sinusoidal position), and merging pads by repeating the last segments even when
fewer segments exist than are needed.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `CrossformerParameters`: segment length, merge size, routers and Transformer sizes. |
| [segment_embedding.py](./segment_embedding.py) | `CrossformerSegmentEmbedding`: dimension-segment-wise embedding with learnable positions. |
| [two_stage_attention_layer.py](./two_stage_attention_layer.py) | `CrossformerTwoStageAttentionLayer`: cross-time attention, then cross-dimension attention through routers. |
| [segment_merging.py](./segment_merging.py) | `CrossformerSegmentMerging`: adjacent segments merged into a coarser scale. |
| [encoder.py](./encoder.py) | `CrossformerEncoder`: embedding and the encoded tokens of every scale. |
| [decoder_layer.py](./decoder_layer.py) | `CrossformerDecoderLayer`: two-stage self-attention, cross-attention to one scale and feed-forward network. |
| [forecast_network.py](./forecast_network.py) | `CrossformerForecastNetwork`: sum of the segment predictions of every decoder layer. |
| [reconstruction_network.py](./reconstruction_network.py) | `CrossformerReconstructionNetwork`: each channel's coarsest tokens to its series. |
| [summary_network.py](./summary_network.py) | `CrossformerSummaryNetwork`: all flattened coarsest tokens to the output vector. |
