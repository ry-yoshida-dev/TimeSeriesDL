# evaluation

## Overview

Measures of how closely synthetic sequences resemble real ones: the discriminative and predictive scores of TimeGAN,
which train small post-hoc networks, and two cheap distribution distances that train nothing. Every measure is seeded
and lower is better.

## Components

| Component | Description |
| --------- | ----------- |
| [settings.py](./settings.py) | `GenerationEvaluationSettings`: post-hoc training, test fraction, split seed, quantiles and lags. |
| [discriminative_score.py](./discriminative_score.py) | `DiscriminativeScore`: `|accuracy - 0.5|` of a real-versus-synthetic classifier on held-out sequences. |
| [predictive_score.py](./predictive_score.py) | `PredictiveScore`: error on real sequences of a next-step predictor trained on synthetic ones. |
| [marginal_distance.py](./marginal_distance.py) | `MarginalDistance`: quantile distance between the per-channel value distributions. |
| [autocorrelation_distance.py](./autocorrelation_distance.py) | `AutocorrelationDistance`: difference of the mean autocorrelation functions. |
| [recurrent_scorer.py](./recurrent_scorer.py) | `RecurrentScorer`: the post-hoc GRU of both scores. |
| [discrimination_step.py](./discrimination_step.py) | `DiscriminationStep`: cross-entropy of the classifier. |
| [next_step_prediction_step.py](./next_step_prediction_step.py) | `NextStepPredictionStep`: absolute error of the predictor. |
