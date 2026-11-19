# state_space

## Overview

Building blocks of the state space models: HiPPO memories, discretization, the parallel scan and FFT convolution
that evaluate linear recurrences, the convolution kernels of S4 and S4D, and the state space layers that the deep
state space encoder stacks.

## Components

| Component | Description |
| --------- | ----------- |
| [hippo_measure.py](./hippo_measure.py) | `HiPPOMeasure`: scaled Legendre, translated Legendre, Laguerre or Fourier dynamics. |
| [hippo_memory.py](./hippo_memory.py) | `HiPPOMemory`: fixed discrete HiPPO recurrence and its impulse response. |
| [discretization.py](./discretization.py) | `Discretization`: zero-order hold or bilinear, for dense and diagonal models. |
| [normal_plus_low_rank.py](./normal_plus_low_rank.py) | `NormalPlusLowRank`: HiPPO-LegS as a diagonalized normal matrix minus a rank-one term. |
| [diagonal_initialization.py](./diagonal_initialization.py) | `DiagonalInitialization`: S4D-LegS, S4D-Inv or S4D-Lin eigenvalues. |
| [linear_recurrence.py](./linear_recurrence.py) | `LinearRecurrence`: parallel scan of a diagonal linear recurrence. |
| [fft_convolution.py](./fft_convolution.py) | `FFTConvolution`: causal convolution with a sequence-long kernel. |
| [causal_depthwise_convolution.py](./causal_depthwise_convolution.py) | `CausalDepthwiseConvolution`: short causal convolution of each channel. |
| [gated_linear_output.py](./gated_linear_output.py) | `GatedLinearOutput`: GELU, dropout and gated linear unit after a state space model. |
| [ssm_kernel.py](./ssm_kernel.py) | `SSMKernel`: convolution kernel of one state space model per channel. |
| [ssm_kernels/](./ssm_kernels/README.md) | Implementations of `SSMKernel`. |
| [state_space_layer.py](./state_space_layer.py) | `StateSpaceLayer`: sequence to sequence of the same size through a state space model. |
| [state_space_layers/](./state_space_layers/README.md) | Implementations of `StateSpaceLayer`. |
