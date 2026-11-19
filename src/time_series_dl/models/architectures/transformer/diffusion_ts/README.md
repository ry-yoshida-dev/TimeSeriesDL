# diffusion_ts

## Overview

Diffusion-TS (Yuan and Qiao, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", ICLR 2024):
a denoising diffusion model whose encoder-decoder Transformer estimates the clean sequence directly, as a polynomial
trend plus Fourier seasonality plus a residual, trained with a time and frequency domain loss. It is a sequence
generation model for the generation task.

Differences from the paper and the authors' code: the diffusion step is embedded once by sinusoidal features and two
SiLU layers shared by every adaptive normalization (the authors give each normalization its own sinusoidal
features); the seasonal components use the amplitude normalization of an inverse discrete Fourier transform, as in
ETSformer; the feed-forward networks use GELU; sampling runs every diffusion step of the ancestral sampler, without
clipping the clean estimate (the authors clip it to [-1, 1] on data min-max scaled to that range, and also offer
fewer-step sampling); and conditional generation (imputation and forecasting by reconstruction guidance) is not
included.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `DiffusionTSParameters`: sizes, layers, trend degree, seasonality, noise schedule and loss. |
| [cosine_noise_schedule.py](./cosine_noise_schedule.py) | `CosineNoiseSchedule`: cosine noise variances; builds the diffusion process. |
| [loss_kind.py](./loss_kind.py) | `DiffusionTSLossKind`: absolute or squared error. |
| [sequence_generation_network.py](./sequence_generation_network.py) | `DiffusionTSSequenceGenerationNetwork`: weighted time and Fourier domain loss, and sampling. |
| [denoiser.py](./denoiser.py) | `DiffusionTSDenoiser`: encoder-decoder Transformer that estimates the clean sequences. |
| [decomposition.py](./decomposition.py) | `DiffusionTSDecomposition`: trend, seasonality and residual of an estimate. |
| [encoder_layer.py](./encoder_layer.py) | `DiffusionTSEncoderLayer`: self-attention and feed-forward network. |
| [decoder_layer.py](./decoder_layer.py) | `DiffusionTSDecoderLayer`: self- and cross-attention, trend and seasonal blocks, and level. |
| [decoder_layer_output.py](./decoder_layer_output.py) | `DiffusionTSDecoderLayerOutput`: tokens, level, trend and seasonality of a decoder layer. |
| [trend_block.py](./trend_block.py) | `DiffusionTSTrendBlock`: polynomial trend from the tokens. |
| [adaptive_layer_norm.py](./adaptive_layer_norm.py) | `DiffusionTSAdaptiveLayerNorm`: layer normalization scaled and shifted by the diffusion step. |
