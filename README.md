# MENT-Torch

This repository will implement the MENT algorithm in PyTorch. The current NumPy version here: https://github.com/austin-hoover/ment.

The only possible advantage of PyTorch is when using a differentiable physics simulation to transform the distribution. In this case, the MENT distribution function $\rho(x)$ is differentiable, and one can try to use generative models to sample particles from the distribution. If this ends up being useful, development may switch to this repository.
