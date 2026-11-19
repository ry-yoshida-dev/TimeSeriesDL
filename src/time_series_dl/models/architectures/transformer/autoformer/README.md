# autoformer

## Overview

Autoformer (Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series
Forecasting", NeurIPS 2021): the auto-correlation mechanism, which aggregates the values at the delays of highest
period-based correlation, in the decomposition Transformer of
[decomposition](../decomposition/README.md). A sinusoidal position encoding replaces the time-stamp
embedding of the paper. Reconstructions and summaries use the encoder and the generic heads.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `AutoformerParameters`: sizes, layers, moving average, delay factor, label length and head. |
