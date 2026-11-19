# objectives

## Overview

Implementations of `ForecastObjective`: what a forecast network predicts for each future value, how it is trained, and
how sample paths are drawn from it.

`LowRankGaussianObjective` works with every forecast network, since each target channel's mean, deviation and factor
row are ordinary per-value parameters (checked with all 58 forecasting architectures). Networks that mix channels
(iTransformer, TSMixer, the graph networks, MLP, RNN and most encoders) can make the correlation depend on the window;
channel-independent networks (PatchTST, DLinear, NLinear, RLinear, FITS, SparseTSF, SegRNN and others that share
weights across channels) compute each channel's factor row from that channel alone, so the learned correlation is
mostly the same in every window. Generative networks need `SampleObjective`.

## Components

| Component | Description |
| --------- | ----------- |
| [point.py](./point.py) | `PointObjective`: one value, trained with any `PointLoss`. |
| [quantile.py](./quantile.py) | `QuantileObjective`: non-crossing quantiles, trained with the pinball loss. |
| [gaussian.py](./gaussian.py) | `GaussianObjective`: mean and standard deviation, trained with the negative log-likelihood. |
| [student_t.py](./student_t.py) | `StudentTObjective`: location, scale and degrees of freedom; heavy-tailed likelihood. |
| [laplace.py](./laplace.py) | `LaplaceObjective`: location and scale; likelihood that estimates the median. |
| [low_rank_gaussian.py](./low_rank_gaussian.py) | `LowRankGaussianObjective`: multivariate normal of the target channels at each step with a low-rank plus diagonal covariance (GPVar). |
| [sample.py](./sample.py) | `SampleObjective`: an ensemble of sample paths, drawn by generative networks or output directly and trained with the CRPS. |
