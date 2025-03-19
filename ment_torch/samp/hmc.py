# https://github.com/AdamCobb/hamiltorch
from enum import Enum
from typing import Callable
from typing import Optional

from numpy import pi
import torch
import torch.nn as nn

from . import hmc_utils as utils


class Sampler(Enum):
    HMC = 1
    HMC_NUTS = 3


class Integrator(Enum):
    EXPLICIT = 1
    IMPLICIT = 2
    S3 = 3
    SPLITTING = 4
    SPLITTING_RAND = 5
    SPLITTING_KMID = 6


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


# def adaptation(rho: float, t: int, step_size_init: float, H_t: float, eps_bar: float, desired_accept_rate: float = 0.8) -> tuple:
#     """No-U-Turn sampler adaptation of the step size. This follows Algo 5, p. 15 from Hoffman and Gelman 2011.
#
#     Parameters
#     ----------
#     rho : float
#         rho is current acceptance ratio.
#     t : int
#         Iteration.
#     step_size_init : float
#         Initial step size.
#     H_t : float
#         Current rolling H_t.
#     eps_bar : type
#         Current rolling step size update.
#     desired_accept_rate : float
#         The step size is adapted with the objective of a desired acceptance rate.
#
#     Returns
#     -------
#     step_size : float
#         Current step size to be used.
#     eps_bar : float
#         Current rolling step size update. Also at last iteration this is the final adapted step size.
#     H_t : float
#         Current rolling H_t to be passed at next iteration.
#     """
#     t = t + 1
#
#     if utils.has_nan_or_inf(torch.tensor([rho])):
#         alpha = 0.0  # Acceptance rate is zero if nan.
#     else:
#         alpha = min(1.0, float(torch.exp(torch.FloatTensor([rho]))))
#
#     mu = float(torch.log(10 * torch.FloatTensor([step_size_init])))
#     gamma = 0.05
#     t0 = 10
#     kappa = 0.75
#     H_t = (1 - (1 / (t + t0))) * H_t + (1 / (t + t0)) * (desired_accept_rate - alpha)
#     x_new = mu - (t**0.5) / gamma * H_t
#     step_size = float(torch.exp(torch.FloatTensor([x_new])))
#     x_new_bar = t**-kappa * x_new + (1 - t**-kappa) * torch.log(torch.FloatTensor([eps_bar]))
#     eps_bar = float(torch.exp(x_new_bar))
#
#     return step_size, eps_bar, H_t


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
    debug: int = 0,
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
    debug : int
        - 0: No output
        - 1: Print old/new Hamiltonian.
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
        if verbose:
            utils.progress_bar_update(n)

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

        if debug == 1:
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

    if verbose:
        utils.progress_bar_end("Acceptance rate = {:.2f}".format(n_accepted / size))

    return ret_params