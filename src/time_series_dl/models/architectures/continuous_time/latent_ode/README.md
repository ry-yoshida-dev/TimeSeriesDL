# latent_ode

## Overview

Latent ODE (Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS 2019). A latent state at time
0 follows a neural ordinary differential equation and a linear map of it gives the mean of the observations; an
ODE-RNN read backward from the last observation to time 0 infers its Gaussian posterior. Interpolation and
forecasting decode the posterior mean at the query times; classification and regression read the initial latent
state with the output layers and add `output_loss_weight` times the task loss to the negative evidence lower bound,
as in the paper's PhysioNet mortality experiment.

Differences from the paper: one latent sample per sequence and a constant `kl_weight` instead of several samples and
KL annealing; the posterior standard deviation is a softplus instead of an absolute value; the recognition ODE-RNN
uses a standard GRU cell; both equations are solved with fixed steps (Euler for the recognition network and the
fifth-order Dormand-Prince tableau for the latent dynamics, as the authors' solvers but without step-size adaptation);
the latent trajectory of each sequence is solved at its own times rather than on the union of the batch's times.

## Components

| Component | Description |
| --------- | ----------- |
| [parameters.py](./parameters.py) | `LatentODEParameters`: sizes, solvers, likelihood, loss weights and head. |
| [latent_posterior.py](./latent_posterior.py) | `LatentPosterior`: Gaussian posterior of the initial latent state. |
| [generative_model.py](./generative_model.py) | `LatentODEGenerativeModel`: recognition, latent dynamics, decoding and evidence lower bound. |
| [irregular_summary_network.py](./irregular_summary_network.py) | `LatentODESummaryNetwork`: output layers on the initial latent state. |
| [irregular_interpolation_network.py](./irregular_interpolation_network.py) | `LatentODEInterpolationNetwork`: the posterior mean decoded at query times. |
