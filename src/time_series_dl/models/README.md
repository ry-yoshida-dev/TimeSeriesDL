# models

## Overview

Parameters and networks of every model. A parameters class builds the networks of the tasks it supports:
forecasting (`build_forecast_network`), classification and regression (`build_summary_network`), and imputation and
anomaly detection (`build_reconstruction_network`). Every public class is imported from `time_series_dl.models`.

| Family | Models |
| ------ | ------ |
| [Linear](./architectures/linear/README.md) | DLinear, NLinear |
| [MLP](./architectures/mlp/README.md) | MLP, N-BEATS, N-HiTS, TSMixer, TiDE |
| [Recurrent](./architectures/recurrent/README.md) | LSTM / GRU / Elman RNN, SegRNN, DeepAR |
| [Convolutional](./architectures/convolutional/README.md) | FCN, ResNet, InceptionTime, TCN, TimesNet |
| [Transformer](./architectures/transformer/README.md) | Transformer, Informer, Autoformer, FEDformer, PatchTST, iTransformer, Temporal Fusion Transformer |

`ModelSettings` names a model by its `ModelArchitecture` and its parameters by field, so that a configuration file
can describe it; the YAML presets of every model are in [config/](../config/README.md).

Implementations follow the papers, with details taken from Time-Series-Library, neuralforecast and the authors' code;
each model's README states where it differs. Time-stamp, static and known-future covariates are not supported.

## Components

| Component | Description |
| --------- | ----------- |
| [interfaces/](./interfaces/README.md) | Abstract networks, builder protocols and shapes that every model and task share. |
| [components/](./components/README.md) | Building blocks shared by several models. |
| [architectures/](./architectures/README.md) | Every model, grouped by family. |
| [settings/](./settings/README.md) | Model selection and parameters from configuration files. |
