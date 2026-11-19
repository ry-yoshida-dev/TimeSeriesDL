# time_mixer_plus_plus

## Overview

TimeMixer++ (Wang et al., "TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis",
ICLR 2025): the window is down-sampled by strided convolutions into several scales, the coarsest scale is mixed across
variates by self-attention, and every scale is embedded into tokens. Each mixer block folds the tokens of every scale
into 2D time images at the top-K periods of the coarsest scale (MRTI), splits each image into seasonal and trend
images by dual-axis attention (TID), mixes seasonal images bottom-up with strided 2D convolutions and trend images
top-down with transposed 2D convolutions (MCM), and sums the periods weighted by the softmax of their amplitudes with
a residual connection (MRM). Forecasting, classification / regression and imputation / anomaly detection.

Choices where the paper leaves room, and differences from it:

- Rows of a time image are the phases within a period and columns the successive periods; the seasonal image comes
  from attention among the phases of each period, the trend image from attention among the periods at each phase.
  As in the paper, the periods `ceil(T / f)` are measured in steps of the input and shared by every scale.
- The down-sampling is one circular convolution (kernel 3, stride 2) over channels applied repeatedly, so each scale
  has `ceil` of half the previous length; the coarsest scale needs at least two steps.
- Channel mixing is one-head attention among variates with a residual connection; the embedding is a circular
  convolution over channels, as in TimeMixer.
- The forecast sums one linear predictor per scale and projects the tokens to every channel (channel mixing rather
  than channel independence); a learnable affine RevIN wraps the network, and only the parameters in the scale of
  the values are denormalized.
- Summary and reconstruction read every scale (flattened tokens, or per-scale maps back to the length), so that each
  parameter is trained, and use no instance normalization.
- The defaults (3 down-samplings, 2 blocks, top-3 periods) follow the long-term forecasting setting; the sizes are
  chosen within the range the paper searches.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `TimeMixerPlusPlusParameters`: sizes, blocks, down-samplings, periods and dropout. |
| [time_imaging.py](./time_imaging.py) | `TimeMixerPlusPlusTimeImaging`: period detection and folding into 2D time images. |
| [image_decomposition.py](./image_decomposition.py) | `TimeMixerPlusPlusImageDecomposition`: dual-axis attention into seasonal and trend images. |
| [scale_mixing.py](./scale_mixing.py) | `TimeMixerPlusPlusScaleMixing`: bottom-up seasonal and top-down trend mixing across scales. |
| [mixer_block.py](./mixer_block.py) | `TimeMixerPlusPlusMixerBlock`: imaging, decomposition, scale mixing and period mixing. |
| [encoder.py](./encoder.py) | `TimeMixerPlusPlusEncoder`: down-sampling, channel attention, embedding and mixer blocks. |
| [forecast_network.py](./forecast_network.py) | `TimeMixerPlusPlusForecastNetwork`: one predictor per scale, summed into the horizon. |
| [reconstruction_network.py](./reconstruction_network.py) | `TimeMixerPlusPlusReconstructionNetwork`: every scale mapped back onto the window. |
| [summary_network.py](./summary_network.py) | `TimeMixerPlusPlusSummaryNetwork`: flattened tokens of every scale to the output vector. |
