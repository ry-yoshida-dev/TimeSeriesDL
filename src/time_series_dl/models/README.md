# models

## Overview

Parameters and networks of every model. A parameters class builds the networks of the tasks it supports:
forecasting (`build_forecast_network`), classification and regression (`build_summary_network`), imputation and
anomaly detection (`build_reconstruction_network`), and imputation from the values with their missing-value mask
(`build_masked_reconstruction_network`). Every public class is imported from `time_series_dl.models`.

| Family | Models |
| ------ | ------ |
| [Linear](./architectures/linear/README.md) | DLinear, NLinear, RLinear, FITS, SparseTSF |
| [MLP](./architectures/mlp/README.md) | MLP, N-BEATS, N-HiTS, TSMixer, TiDE, TimeMixer, TimeMixer++, FreTS, Koopa |
| [Recurrent](./architectures/recurrent/README.md) | LSTM / GRU / Elman RNN, SegRNN, DeepAR, xLSTM, xLSTMTime, minGRU / minLSTM, LSTNet, BRITS, GRU-D |
| [Convolutional](./architectures/convolutional/README.md) | FCN, ResNet, InceptionTime, TCN, TimesNet, MICN, ModernTCN, LITE, ROCKET, MiniRocket, MultiRocket |
| [Transformer](./architectures/transformer/README.md) | Transformer, Informer, Autoformer, FEDformer, PatchTST, iTransformer, Temporal Fusion Transformer, Non-stationary Transformer, Pyraformer, ETSformer, Crossformer, TimeXer, TST, ConvTran, SAITS |
| [State space](./architectures/state_space/README.md) | HiPPO-RNN, LMU, S4, S4D, S5, LRU, Mamba, Mamba-2, S-Mamba, FiLM |
| [Graph](./architectures/graph/README.md) | MTGNN, Graph WaveNet, AGCRN, StemGNN, FourierGNN |

`ModelSettings` names a model by its `ModelArchitecture` and its parameters by field, so that a configuration file
can describe it; the YAML presets of every model are in [config/](../config/README.md).

Implementations follow the papers, with details taken from Time-Series-Library, neuralforecast and the authors' code;
each model's README states where it differs. Forecasts accept known-future, static numeric and static categorical covariates: TiDE, Temporal Fusion
Transformer and DeepAR read them as in their papers (`CovariateForecastingModel`), and every other model through
the generic covariate adapter in [components/covariates/](./components/covariates/README.md).

## Components

| Component | Description |
| --------- | ----------- |
| [interfaces/](./interfaces/README.md) | Abstract networks, builder protocols and shapes that every model and task share. |
| [components/](./components/README.md) | Building blocks shared by several models. |
| [architectures/](./architectures/README.md) | Every model, grouped by family. |
| [settings/](./settings/README.md) | Model selection and parameters from configuration files. |
