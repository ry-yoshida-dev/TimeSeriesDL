# time_grad

## Overview

TimeGrad (Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series
Forecasting", ICML 2021): recurrent layers summarize the series up to each step, and a denoising diffusion model
conditioned on that summary draws the next multivariate step; sample paths are drawn autoregressively. It is a
generative model: it is trained with the denoising loss and forecasts with `SampleObjective`, and reads no covariates.

Differences from the paper and the authors' PyTorchTS code: the task normalizes each window instead of mean scaling;
the inputs are the past window's channels and then the previous targets, without lag features, time features or
target dimension embeddings; the denoiser's convolutions over the target dimensions use zero "same" padding instead of
circular padding, which also works for one or two targets; and the conditioning upsampler's hidden size is at least 1.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimeGradParameters`: recurrent layers, conditioning size, noise schedule and denoiser sizes. |
| [generative_forecast_network.py](./generative_forecast_network.py) | `TimeGradForecastNetwork`: denoising loss of every step after the first, and autoregressive sampling of the horizon. |
| [denoiser.py](./denoiser.py) | `TimeGradDenoiser`: predicts the noise of one step from the diffusion step and the hidden state. |
| [residual_block.py](./residual_block.py) | `TimeGradResidualBlock`: gated dilated convolution over the target dimensions. |
