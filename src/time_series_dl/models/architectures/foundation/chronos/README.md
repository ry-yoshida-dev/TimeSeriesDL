# chronos

## Overview

Chronos (Ansari et al., "Chronos: Learning the Language of Time Series", TMLR 2024): a T5 encoder-decoder language
model pretrained on tokens of time series. Each series is divided by the mean absolute value of its context and
quantized into 4093 uniform bins in [-15, 15]; the encoder reads the context tokens followed by EOS, and the decoder
samples the next tokens, which map back to bin centers times the scale. Every target channel is forecast as a
separate univariate series. The network is a `GenerativeForecastNetwork`: it outputs sample paths, which
`SampleObjective` summarizes (the official pipeline draws 20), and fine-tuning minimizes the token cross-entropy of
the true future values with teacher forcing, as in the paper's training.

`ChronosParameters.pretrained(ChronosSize.SMALL)` loads `amazon/chronos-t5-{tiny,mini,small,base,large}` (Apache-2.0)
at a pinned revision; `Forecaster.prepare_zero_shot` forecasts with it without training. Chronos's mean scaling is
not shift invariant, so use `NoScalingParameters()` and `is_instance_normalized=False` to feed it the raw values it
was pretrained on.

Differences from the official code: sampling reimplements the `transformers` generation the pipeline uses (EOS
suppressed because exactly the requested number of tokens is generated, then temperature, top-k and top-p), so
sample paths have the same distribution but not the same random draws. Fine-tuning over horizons longer than
`prediction_length` teacher-forces the whole horizon in one sequence, which the published models were not trained on.
The causal (GPT-2) variant of the paper is not provided; every published checkpoint is the T5 variant with the output
layer tied to the token embedding.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `ChronosParameters`: T5 sizes, tokenization, generation settings and published weights. |
| [size.py](./size.py) | `ChronosSize`: the published checkpoints and their pinned revisions. |
| [tokenizer.py](./tokenizer.py) | `ChronosTokenizer`: mean scaling and uniform quantization into tokens, and back. |
| [model.py](./model.py) | `ChronosModel`: encoder-decoder over tokens, token cross-entropy and sampling of future paths. |
| [forecast_network.py](./forecast_network.py) | `ChronosForecastNetwork`: each target channel as a univariate series, sampled or trained. |
| [official_names.py](./official_names.py) | `ChronosOfficialNames`: the published `transformers` parameter names. |
