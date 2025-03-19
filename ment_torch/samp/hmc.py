# https://github.com/AdamCobb/hamiltorch
from enum import Enum
from typing import Callable
from typing import Optional

from numpy import pi
import torch
import torch.nn as nn


def compute_gradients_loop(log_prob_func: Callable, x: torch.Tensor) -> torch.Tensor:
    """Vectorized gradient calculation (for loop)."""
    x = x.detach().requires_grad_()
    log_prob = log_prob_func(x)
    for i in range(x.shape[0]):
        log_prob[i].backward(retain_graph=True)
    return x.grad


def compute_gradients(log_prob_func: Callable, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_()
    log_prob = log_prob_func(x)
    log_prob.sum().backward()
    return x.grad


def resample_momentum(params: torch.Tensor, cov_matrix: torch.Tensor = None) -> torch.Tensor:
    """Resample momentum distribution."""
    mean = torch.zeros(params.shape[1])
    if cov_matrix is None:
        cov_matrix = torch.eye(params.shape[1])
    dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
    return dist.sample((params.shape[0],))


def leapfrog(
    x: torch.Tensor,
    p: torch.Tensor,
    log_prob_func: Callable,
    steps: int = 10,
    step_size: float = 0.1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Propose new set of parameters and momentum.

    Parameters
    ----------
    x : torch.tensor
        Parameter array, shape (..., D).
    p : torch.tensor
        Momentum array, shape (..., D).
    log_prob_func : function
        Vectorized log-probability function.
    steps : int
        Number of steps per trajectory (often referred to as L).
    step_size : float
        Integration step size.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC.
        Choices = {
            Sampler.HMC,
            Sampler.HMC_NUTS,
        }

    Returns
    -------
    ret_params : list[torch.Tensor]
        List of parameters collected in the trajectory.
    ret_momentum : list[torch.Tensor]
        List of momentum collected in the trajectory.
    """
    x = torch.clone(x)
    p = torch.clone(p)
    
    xs, ps = [], []

    p += 0.5 * step_size * compute_gradients(log_prob_func, x)
    for _ in range(steps):
        x += step_size * p
        p += step_size * compute_gradients(log_prob_func, x)
        xs.append(torch.clone(x.detach()))
        ps.append(torch.clone(p.detach()))
    # Only need last for Hamiltonian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
    ps[-1] -= 0.5 * step_size * compute_gradients(log_prob_func, x)

    return xs, ps


def hamiltonian(params: torch.Tensor, momentum: torch.Tensor, log_prob_func: Callable) -> torch.Tensor:
    potential = -log_prob_func(params)
    kinetic = 0.5 * torch.sum(torch.square(momentum), axis=1)
    return potential + kinetic


def sample(
    log_prob_func: Callable,
    start: torch.Tensor,
    size: int = 10,
    steps_per_samp: int = 10,
    step_size: float = 0.1,
    burn: int = 0,
    merge: bool = True,
    verbose: int = 0,

) -> tuple:
    """Vectorized Hamiltonian Monte Carlo.

    Parameters
    ----------
    log_prob_func : Callable
        Vectorized log-probability function.
    start : torch.Tensor
        Initial particle coordinates. If shape=(ndim,), run a single chain;
        otherwise if shape=(nchain, ndim), run multiple chains.
    size : int
        Sets the number of samples corresponding to the number of momentum resampling 
        steps/the number of trajectories to sample.
    steps_per_samp : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    burn : int
        Number of samples to burn before collecting samples. Set to -1 for no burning
        of samples. This must be less than `size` as `size` subsumes `burn`.
    merge: bool
        Whether to merge chains.
    verbose : int
        If 0, do not display progress bar.

    Returns
    -------
    list[torch.Tensor]
        A list of parameter samples.
    """
    device = start.device

    if burn >= size:
        raise RuntimeError("burn must be less than size.")

    if start.ndim == 1:
        start = start[None, :]

    params = start.clone()
    ret_params = [params.clone().detach()]

    n_chains = start.shape[0]
    n_accepted = 0

    for n in range(size + burn):
        # Push particles
        momentum = resample_momentum(params)
        ham = hamiltonian(params, momentum, log_prob_func)
        leapfrog_params, leapfrog_momenta = leapfrog(
            params,
            momentum,
            log_prob_func,
            steps=steps_per_samp,
            step_size=step_size,
        )

        # MH proposal
        new_params = leapfrog_params[-1].to(device).detach()
        new_momentum = leapfrog_momenta[-1].to(device)
        new_ham = hamiltonian(new_params, new_momentum, log_prob_func)

        if verbose > 1:
            print("Step: {},".format(n))
            print("Hamiltonian current : {},".format(ham))
            print("Hamiltonian proposed: {},".format(new_ham))

        rho = torch.minimum(torch.zeros(n_chains), ham - new_ham)
        accept = rho >= torch.log(torch.rand(n_chains))
        if n > burn:
            n_accepted += torch.sum(accept)
        params[accept] = new_params[accept]
        ret_params.append(params.detach().clone())

    ret_params = ret_params[burn:]
    ret_params = ret_params[:size]

    if merge:
        ret_params = torch.vstack(ret_params)

    return torch.squeeze(ret_params)