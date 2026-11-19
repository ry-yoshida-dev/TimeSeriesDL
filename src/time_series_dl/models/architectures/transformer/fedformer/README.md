# fedformer

## Overview

FEDformer (Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
ICML 2022): Fourier-enhanced blocks and cross-attention on a random or lowest subset of frequencies, in the
decomposition Transformer of [decomposition](../decomposition/README.md). Only the Fourier
version is implemented, not the wavelet one, and the mixture-of-experts decomposition is simplified to a single moving
average, and a sinusoidal position encoding replaces the time-stamp embedding. Reconstructions and summaries use the
encoder and the generic heads.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `FEDformerParameters`: sizes, layers, moving average, Fourier modes, label length and head. |
