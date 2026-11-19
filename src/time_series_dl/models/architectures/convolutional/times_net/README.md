# times_net

## Overview

TimesNet (Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023): the
sequence is folded into 2D grids at its dominant Fourier periods, and inception convolutions model the variation
within and across periods.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimesNetParameters`: sizes, number of periods and kernels, and dropout. |
| [period_detector.py](./period_detector.py) | `PeriodDetector`: dominant periods and their amplitudes from the Fourier spectrum. |
| [inception_block.py](./inception_block.py) | `InceptionBlock`: average of 2D convolutions of several kernel sizes. |
| [times_block.py](./times_block.py) | `TimesBlock`: 2D variation at each period, aggregated by amplitude. |
| [embedding.py](./embedding.py) | `TimesNetEmbedding`: convolutional value embedding with position encoding. |
| [forecast_network.py](./forecast_network.py) | `TimesNetForecastNetwork`: window extended over the horizon, then TimesBlocks. |
| [reconstruction_network.py](./reconstruction_network.py) | `TimesNetReconstructionNetwork`: TimesBlocks on the sequence, projected to its values. |
| [summary_network.py](./summary_network.py) | `TimesNetSummaryNetwork`: TimesBlocks, then the flattened features to the output vector. |
