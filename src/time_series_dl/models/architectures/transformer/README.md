# transformer

## Overview

Models built from attention or its replacements.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation / anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :----------------------------: |
| [Transformer (Vaswani et al., 2017)](./vanilla/README.md) | `TransformerParameters` | ✓ | ✓ | ✓ |
| [Informer (Zhou et al., 2021)](./informer/README.md) | `InformerParameters` | ✓ | ✓ | ✓ |
| [Autoformer (Wu et al., 2021)](./autoformer/README.md) | `AutoformerParameters` | ✓ | ✓ | ✓ |
| [FEDformer (Zhou et al., 2022)](./fedformer/README.md) | `FEDformerParameters` | ✓ | ✓ | ✓ |
| [PatchTST (Nie et al., 2023)](./patch_tst/README.md) | `PatchTSTParameters` | ✓ | ✓ | ✓ |
| [iTransformer (Liu et al., 2024)](./itransformer/README.md) | `ITransformerParameters` | ✓ | ✓ | ✓ |
| [Temporal Fusion Transformer (Lim et al., 2021)](./temporal_fusion_transformer/README.md) | `TemporalFusionTransformerParameters` | ✓ | | |
| [Decomposition Transformer](./decomposition/README.md) | Built by `AutoformerParameters` and `FEDformerParameters` | ✓ | ✓ | ✓ |
