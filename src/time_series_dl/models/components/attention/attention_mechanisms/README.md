# attention_mechanisms

## Overview

Implementations of `AttentionMechanism`, from the vanilla Transformer (Vaswani et al., 2017), Informer (Zhou et al.,
AAAI 2021), Autoformer (Wu et al., NeurIPS 2021) and FEDformer (Zhou et al., ICML 2022).

## Components

| Component | Description |
| --------- | ----------- |
| [full.py](./full.py) | `FullAttentionMechanism`: scaled dot-product attention, optionally causal. |
| [prob_sparse.py](./prob_sparse.py) | `ProbSparseAttentionMechanism`: Informer's attention of the most active queries only. |
| [auto_correlation.py](./auto_correlation.py) | `AutoCorrelationAttentionMechanism`: Autoformer's time-delay aggregation. |
| [fourier.py](./fourier.py) | `FourierAttentionMechanism`: FEDformer's Fourier-enhanced block. |
| [fourier_cross.py](./fourier_cross.py) | `FourierCrossAttentionMechanism`: FEDformer's Fourier-enhanced cross-attention. |
