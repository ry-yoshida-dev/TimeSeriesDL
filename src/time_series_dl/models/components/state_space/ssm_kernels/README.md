# ssm_kernels

## Overview

Implementations of `SSMKernel`, the convolution kernels of linear time-invariant state space models.

## Components

| Component | Description |
| --------- | ----------- |
| [dplr.py](./dplr.py) | `DPLRSSMKernel`: S4's diagonal plus low-rank state matrix, through Cauchy products at the roots of unity. |
| [diagonal.py](./diagonal.py) | `DiagonalSSMKernel`: S4D's complex diagonal state matrix, through a Vandermonde product. |
