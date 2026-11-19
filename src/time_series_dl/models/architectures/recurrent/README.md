# recurrent

## Overview

Models built from recurrent layers.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation / anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :----------------------------: |
| [LSTM / GRU / Elman RNN](./vanilla/README.md) | `RNNParameters` | ✓ | ✓ | ✓ |
| [SegRNN (Lin et al., 2023)](./seg_rnn/README.md) | `SegRNNParameters` | ✓ | | |
| [DeepAR (Salinas et al., 2020)](./deep_ar/README.md) | `DeepARParameters` | ✓ | | |
