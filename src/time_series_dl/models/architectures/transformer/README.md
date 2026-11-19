# transformer

## Overview

Models built from attention or its replacements.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation | Anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :--------: | :---------------: |
| [Transformer (Vaswani et al., 2017)](./vanilla/README.md) | `TransformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [Informer (Zhou et al., 2021)](./informer/README.md) | `InformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [Autoformer (Wu et al., 2021)](./autoformer/README.md) | `AutoformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [FEDformer (Zhou et al., 2022)](./fedformer/README.md) | `FEDformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [PatchTST (Nie et al., 2023)](./patch_tst/README.md) | `PatchTSTParameters` | ✓ | ✓ | ✓ | ✓ |
| [iTransformer (Liu et al., 2024)](./itransformer/README.md) | `ITransformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [Temporal Fusion Transformer (Lim et al., 2021)](./temporal_fusion_transformer/README.md) | `TemporalFusionTransformerParameters` | ✓ | | | |
| [Non-stationary Transformer (Liu et al., 2022)](./non_stationary_transformer/README.md) | `NonStationaryTransformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [Pyraformer (Liu et al., 2022)](./pyraformer/README.md) | `PyraformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [ETSformer (Woo et al., 2022)](./etsformer/README.md) | `ETSformerParameters` | ✓ | ✓ | | |
| [Crossformer (Zhang & Yan, 2023)](./crossformer/README.md) | `CrossformerParameters` | ✓ | ✓ | ✓ | ✓ |
| [TimeXer (Wang et al., 2024)](./time_xer/README.md) | `TimeXerParameters` | ✓ | | | |
| [TST (Zerveas et al., 2021)](./tst/README.md) | `TSTParameters` | ✓ | ✓ | ✓ | ✓ |
| [ConvTran (Foumani et al., 2024)](./conv_tran/README.md) | `ConvTranParameters` | ✓ | ✓ | ✓ | ✓ |
| [SAITS (Du et al., 2023)](./saits/README.md) | `SAITSParameters` | | | ✓ | |
| [Anomaly Transformer (Xu et al., 2022)](./anomaly_transformer/README.md) | `AnomalyTransformerParameters` | | | | ✓ |
| [DCdetector (Yang et al., 2023)](./dcdetector/README.md) | `DCdetectorParameters` | | | | ✓ |
| [TranAD (Tuli et al., 2022)](./tranad/README.md) | `TranADParameters` | | | | ✓ |
| [CSDI (Tashiro et al., 2021)](./csdi/README.md) | `CSDIParameters` | ✓ (generative) | | ✓ | |
| [Diffusion-TS (Yuan and Qiao, 2024)](./diffusion_ts/README.md), generation only | `DiffusionTSParameters` | | | | |
| [Decomposition Transformer](./decomposition/README.md) | Built by `AutoformerParameters` and `FEDformerParameters` | ✓ | ✓ | ✓ | ✓ |
