# MENT-Torch

This repository will implement the MENT algorithm in PyTorch. The current NumPy version here: https://github.com/austin-hoover/ment.

A key step in MENT is to compute the projections of a distribution function $\rho(x)$. We estimate these integrals by sampling particles from $\rho(x)$. In high-dimensional (6D) problems, gridless sampling methods like MCMC are required. With a differentiable particle tracking model, one can explore sampling methods that use the gradient of the distribution function. If this proves useful, development may switch to this repo. 

