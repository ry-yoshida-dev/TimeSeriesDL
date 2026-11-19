# recurrent

## Overview

Models built from recurrent layers.

## Components

| Component | Parameters | Forecasting | Classification / regression | Imputation / anomaly detection |
| --------- | ---------- | :---------: | :-------------------------: | :----------------------------: |
| [LSTM / GRU / Elman RNN](./vanilla/README.md) | `RNNParameters` | ✓ | ✓ | ✓ |
| [SegRNN (Lin et al., 2023)](./seg_rnn/README.md) | `SegRNNParameters` | ✓ | | |
| [DeepAR (Salinas et al., 2020)](./deep_ar/README.md) | `DeepARParameters` | ✓ | | |
| [xLSTM (Beck et al., 2024)](./xlstm/README.md) | `XLSTMParameters` | ✓ | ✓ | ✓ |
| [xLSTMTime (Alharthi and Mahmood, 2024)](./xlstm_time/README.md) | `XLSTMTimeParameters` | ✓ | | |
| [minGRU / minLSTM (Feng et al., 2024)](./min_rnn/README.md) | `MinRNNParameters` | ✓ | ✓ | ✓ |
| [LSTNet (Lai et al., 2018)](./lstnet/README.md) | `LSTNetParameters` | ✓ | ✓ | |
