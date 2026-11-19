# pyraformer

## Overview

Pyraformer (Liu et al., "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and
Forecasting", ICLR 2022): a coarser-scale construction module builds a pyramid of the embedded series with strided
convolutions, pyramidal attention lets each node attend only to its neighbours within its scale, its parent and its
children, and every step is represented by the concatenated features of its ancestors at all scales. Forecasts map
the last step's features to the horizon; reconstructions map each step's features to its values; summaries flatten
every step's features, as in the Time-Series-Library.

The pyramidal attention is computed densely with a boolean mask instead of the paper's custom CUDA kernel, which
gives the same outputs at quadratic cost. Unlike the Time-Series-Library, the attention keeps the residual connection
and layer normalization of the official code, the forecast covers only the target channels and every distribution
parameter, and inputs shorter than the product of the window sizes are padded at the start with their first step so
that no scale is empty.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `PyraformerParameters`: window sizes, inner size, bottleneck and Transformer sizes. |
| [pyramid.py](./pyramid.py) | `PyraformerPyramid`: scale sizes, pyramidal mask and ancestor indexes of each step. |
| [pyramidal_attention_mechanism.py](./pyramidal_attention_mechanism.py) | `PyraformerPyramidalAttentionMechanism`: attention restricted by the pyramidal mask. |
| [coarser_scale_construction.py](./coarser_scale_construction.py) | `PyraformerCoarserScaleConstruction`: bottleneck convolutions that build the coarser scales. |
| [encoder_layer.py](./encoder_layer.py) | `PyraformerEncoderLayer`: pyramidal attention and feed-forward network. |
| [encoder.py](./encoder.py) | `PyraformerEncoder`: embedding, pyramid, attention layers and per-step features of all scales. |
| [forecast_network.py](./forecast_network.py) | `PyraformerForecastNetwork`: last step's features to the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `PyraformerReconstructionNetwork`: each step's features to its values. |
| [summary_network.py](./summary_network.py) | `PyraformerSummaryNetwork`: all flattened step features to the output vector. |
