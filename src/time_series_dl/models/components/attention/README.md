# attention

## Overview

Building blocks of the attention-based models: multi-head attention around a pluggable attention mechanism (full,
ProbSparse, auto-correlation or Fourier-enhanced), and the token embedding of the values.

## Components

| Component | Description |
| --------- | ----------- |
| [attention_mechanism.py](./attention_mechanism.py) | `AttentionMechanism`: mixing of the values within each head. |
| [attention_mechanisms/](./attention_mechanisms/README.md) | Implementations of `AttentionMechanism`. |
| [fourier_mode_selection.py](./fourier_mode_selection.py) | `FourierModeSelection`: random or lowest frequencies of the Fourier mechanisms. |
| [multi_head_attention.py](./multi_head_attention.py) | `MultiHeadAttention`: head projections around a mechanism. |
| [value_embedding.py](./value_embedding.py) | `ValueEmbedding`: convolutional token embedding with position encoding. |
