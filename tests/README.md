# tests

## Overview

Unit tests of every task, model, objective and utility. They run on the CPU in seconds with small networks and
synthetic series.

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Fixtures: fast training settings and a synthetic multivariate series. |
| [network_checks.py](./network_checks.py) | Shape and gradient checks of the networks of every model. |
| [covariate_network_checks.py](./covariate_network_checks.py) | Shape, gradient and covariate-sensitivity checks of covariate forecast networks. |
| [graph_network_checks.py](./graph_network_checks.py) | Target-channel consistency checks of the graph neural networks. |
| [anomaly_network_checks.py](./anomaly_network_checks.py) | Score, loss and gradient checks of anomaly scoring networks. |
| [irregular_network_checks.py](./irregular_network_checks.py) | Shape, gradient and padding-invariance checks of irregular networks. |
| [irregular_datasets.py](./irregular_datasets.py) | Synthetic irregularly sampled datasets of the irregular task tests. |
| [generative_network_checks.py](./generative_network_checks.py) | Loss, sampling and gradient checks of generative forecast networks. |
| [t5_published_names.py](./t5_published_names.py) | T5 state dicts under the published parameter names, for weight-loading tests. |
| [published_mtan.py](./published_mtan.py) | mTAN parameters that name published weights, for restoring tests of the irregular tasks. |
| [published_ts2vec.py](./published_ts2vec.py) | TS2Vec parameters that name published weights, for restoring tests of representation learning. |
| [generation_network_checks.py](./generation_network_checks.py) | Loss, sampling and gradient checks of sequence generation networks. |
| [masked_network_checks.py](./masked_network_checks.py) | Shape, gradient and mask-semantics checks of mask-aware imputation networks. |
| `test_model_*.py` | Network checks and model-specific behavior of each model. |
| [test_state_space_components.py](./test_state_space_components.py) | HiPPO memories, kernels, scans and convolutions of the state space models against their recurrences. |
| [test_reference_models.py](./test_reference_models.py) | Network checks of MLP, RNN, TCN, DLinear and NLinear. |
| [test_model_settings.py](./test_model_settings.py) | `ModelSettings` and the YAML presets of every model. |
| [test_forecaster.py](./test_forecaster.py) | Forecasting: objectives, target channels, scaling and validation. |
| [test_generative_forecasting.py](./test_generative_forecasting.py) | Forecasting with sample-path models and the sample objective. |
| [test_conformal.py](./test_conformal.py) | Split and adaptive conformal intervals and their coverage. |
| [test_diffusion.py](./test_diffusion.py) | Noise schedules, the forward process and the ancestral sampler. |
| [test_restoring_published_weights.py](./test_restoring_published_weights.py) | Every task reloads a saved model with published weights without downloading them. |
| [test_forecaster_zero_shot.py](./test_forecaster_zero_shot.py) | Zero-shot forecasting and fine-tuning with published weights, without downloads. |
| [test_pretrained_weights.py](./test_pretrained_weights.py) | Strict loading of published weights and the optional Hugging Face download. |
| [test_t5.py](./test_t5.py) | T5 encoder and decoder stacks against explicit computations. |
| [test_forecaster_covariates.py](./test_forecaster_covariates.py) | Forecasting with known-future and static covariates, end to end. |
| [test_covariate_adapter.py](./test_covariate_adapter.py) | The generic covariate adapter of forecast networks. |
| [test_graph_components.py](./test_graph_components.py) | Graph learning layers and graph convolutions against explicit references. |
| [test_classifier.py](./test_classifier.py) | Classification of sequences. |
| [test_regressor.py](./test_regressor.py) | Regression of one or several outputs. |
| [test_imputer.py](./test_imputer.py) | Imputation of missing values. |
| [test_missing_value_components.py](./test_missing_value_components.py) | Time gaps, temporal decay and last observation against their recursions. |
| [test_anomaly_detector.py](./test_anomaly_detector.py) | Anomaly scores, detection and metrics. |
| [test_objectives.py](./test_objectives.py) | Forecast objectives and metrics. |
| [test_training.py](./test_training.py) | Trainer, optimizers, schedules and early stopping. |
| [test_persistence.py](./test_persistence.py) | Saving and loading every task and model, and rejecting invalid checkpoints. |
| [test_data.py](./test_data.py) | Windows, series validation and channel scaling. |
| [test_irregular_data.py](./test_irregular_data.py) | Irregular sequences: validation, padding and scaling. |
| [test_differential_equations.py](./test_differential_equations.py) | ODE solvers against analytic solutions and cubic paths against exact interpolation. |
| [test_irregular_classifier.py](./test_irregular_classifier.py) | Classification of irregularly sampled sequences. |
| [test_irregular_regressor.py](./test_irregular_regressor.py) | Regression of irregularly sampled sequences. |
| [test_irregular_interpolator.py](./test_irregular_interpolator.py) | Interpolation and forecasting of irregularly sampled sequences at query times. |
| [test_sequence_generator.py](./test_sequence_generator.py) | Generation of synthetic sequences, its metrics, and saving and loading. |
| [test_representation_learning.py](./test_representation_learning.py) | Self-supervised pretraining, representations and transfer of pretrained encoders. |
| [test_masked_modeling.py](./test_masked_modeling.py) | Random masks and the decoder of masked-modeling pretraining. |
