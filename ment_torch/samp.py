import math
import time
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import torch
import zuko
from tqdm import tqdm

from .utils import edges_to_coords
from .utils import coords_to_edges
from .utils import get_grid_points


def tqdm_wrapper(iterable, verbose=False):
    return tqdm(iterable) if verbose else iterable


def random_uniform(lb: float, ub: float, size: int, rng: torch.Generator = None, device: torch.device = None) -> torch.Tensor:
    return lb + (ub - lb) * torch.rand(size, device=device, generator=rng)


def random_choice(items: torch.Tensor, size: int, pdf: torch.Tensor, rng: torch.Generator = None) -> torch.Tensor:
    idx = torch.multinomial(items, size, replacement=True, generator=rng)
    return items[idx]


def sample_hist_bins(values: torch.Tensor, size: int, rng: torch.Generator = None) -> torch.Tensor:
    pdf = torch.ravel(values) + 1.00e-15
    pdf = pdf / torch.sum(pdf)
    idx = torch.squeeze(torch.nonzero(pdf))
    idx = random_choice(idx, size, pdf=pdf, rng=rng)
    return idx


def sample_hist(
    values: torch.Tensor,
    edges: List[torch.Tensor],
    size: int,
    noise: float = 0.0,
    device: torch.device = None,
    rng: torch.Generator = None,
) -> torch.Tensor:
    
    ndim = values.ndim
    if ndim == 1:
        edges = [edges]

    idx = sample_hist_bins(values, size, rng=rng)
    idx = torch.unravel_index(idx, values.shape)

    x = torch.zeros((size, ndim), device=device)
    for axis in range(ndim):
        lb = edges[axis][idx[axis]]
        ub = edges[axis][idx[axis] + 1]
        x[:, axis] = random_uniform(lb, ub, size, rng=rng, device=device)
        if noise:
            delta = ub - lb
            x[:, axis] += 0.5 * random_uniform(-delta, delta, size, rng=rng, device=device)
            
    return torch.squeeze(x)


def sample_metropolis_hastings(
    prob_func: Callable,
    ndim: int,
    size: int,
    chains: int = 1,
    burnin: int = 10_000,
    start: torch.Tensor = None,
    proposal_cov: torch.Tensor = None,
    merge: bool = True,
    seed: int = None,
    verbose: int = 0,
    device: torch.device = None,
) -> np.ndarray:
    """Vectorized Metropolis-Hastings.

    https://colindcarroll.com/2019/08/18/very-parallel-mcmc-sampling/

    Parameters
    ----------
    prob_func : Callable
        Function returning probability density p(x) at points x. The function must be
        vectorized so that x is a batch of points of shape (nchains, ndim).
    size : int
        The number of samples per chain (excluding burn-in).
    chains : int
        Number of sampling chains.
    burnin : int
        Number of burnin iterations (applies to each chain).
    start : torch.Tensor
        An array of shape (chains, ndim) giving the starting point of each chain. All
        start points must be in regions of nonzero probability density.
    proposal_cov : torch.Tensor
        We use a Gaussian proposal distribution centered on the current point in
        the random walk. This variable is the covariance matrix of the Gaussian
        distribution.
    merge : bool
        Whether to merge the sampling chains. If the chains are merged, the return
        array has shape (size * chains, ndim). Otherwise if has shape (size, chains, ndim).
    seed : int
        Seed used in random number generators.

    Returns
    -------
    torch.Tensor
        Sampled points with burn-in points discarded. Shape is (size, ndim) if merge=True
        or (size * chains, chains, ndim) if merge=False.
    """      
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        
    # Initialize list of points. From now on we each "point" is really a batch of 
    # size (nchains, ndim). Burnin-points will be discarded later.
    size = size + burnin
    points = torch.zeros((size, chains, ndim), device=device) 

    # Sample proposal points from a Gaussian distribution. (The means will be updated
    # during the random walk.)
    proposal_mean = torch.zeros(ndim, device=device)
    if proposal_cov is None:
        proposal_cov = torch.eye(ndim, device=device)
    proposal_dist = torch.distributions.MultivariateNormal(proposal_mean, proposal_cov)
    proposal_points = proposal_dist.sample((size - 1, chains))

    # Set starting point for each chain. If none is provided, sample from the proposal
    # distribution centered at the origin.
    if start is None:
        start_dist = torch.distributions.MultivariateNormal(proposal_mean, proposal_cov)
        start = start_dist.sample((chains,))
        start *= 0.50
        
    points[0] = start

    # Execute random walks
    random_uniforms = random_uniform(0.0, 1.0, size=(size - 1, chains), rng=rng, device=device)
    accept = torch.zeros(chains, device=device)

    results = {}
    results["n_total_accepted"] = 0
    results["n_total"] = 0
    results["acceptance_rate"] = None

    for i in tqdm_wrapper(range(1, size), verbose):
        proposal_point = points[i - 1] + proposal_points[i - 1]
        proposal_prob = prob_func(proposal_point)
        accept = proposal_prob > prob * random_uniforms[i - 1]

        if i > burnin:
            results["n_total_accepted"] += torch.count_nonzero(accept)
            results["n_total"] += chains
            results["acceptance_rate"] = results["n_total_accepted"] / results["n_total"]
            if verbose > 2:
                print(f"debug {i:05.0f}")
                print(results)

        points[i] = points[i - 1]
        points[i][accept] = proposal_point[accept]
        prob[accept] = proposal_prob[accept]

    points = points[burnin:]

    if verbose > 1:        
        print("debug acceptance rate =", acceptance_rate)
        for axis in range(ndim):
            x_chain_stds = [torch.std( chain[:, axis]) for chain in points]
            x_chain_avgs = [torch.mean(chain[:, axis]) for chain in points]
            print(f"debug axis={axis} between-chain avg(x_chain_std) =", torch.mean(x_chain_stds))
            print(f"debug axis={axis} between-chain std(x_chain_std) =", torch.std( x_chain_stds))
            print(f"debug axis={axis} between-chain avg(x_chain_avg) =", torch.mean(x_chain_avgs))
            print(f"debug axis={axis} between-chain std(x_chain_avg) =", torch.std( x_chain_avgs))
    
            x_avg = torch.mean(torch.hstack([chain[:, axis] for chain in points]))
            x_std = torch.std( torch.hstack([chain[:, axis] for chain in points]))
            print(f"debug axis={axis} x_std =", x_std)
            print(f"debug axis={axis} x_avg =", x_avg)

    if merge:
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
    
    return points


class Sampler:
    def __init__(self, ndim: int, verbose: int = 0) -> None:
        self.ndim = ndim
        self.verbose = verbose

    def __call__(self, prob_func: Callable) -> torch.Tensor:
        raise NotImplementedError        


class GridSampler(Sampler):
    def __init__(
        self,
        limits: list[tuple[float]],
        shape: tuple[int],
        noise: float = 0.0,
        store: bool = True,
        device=None,
        **kws
    ) -> None:
        super().__init__(self, **kws)
        
        self.device = device
        self.shape = shape
        self.limits = limits
        self.ndim = len(limits)
        self.noise = noise
        self.store = store

        self.edges = [
            torch.linspace(
                self.limits[axis][0],
                self.limits[axis][1],
                self.shape[axis] + 1,
            )
            for axis in range(self.ndim)
        ]
        self.coords = [edges_to_coords(e) for e in self.edges]
        self.points = None

    def get_grid_points(self) -> torch.Tensor:
        if self.points is not None:
            return self.points
        points = get_grid_points(self.coords)
        if self.store:
            self.points = points
        return points

    def __call__(self, prob_func: Callable, size: int) -> torch.Tensor:
        prob = prob_func(self.get_grid_points())
        prob = torch.reshape(prob, self.shape)
        x = sample_hist(prob, self.edges, size=size, noise=self.noise, device=self.device)
        return x


class FlowSampler(Sampler):
    def __init__(
        self, 
        flow: zuko.flows.Flow, 
        unnorm_matrix: torch.Tensor = None, 
        train_kws: dict = None,
        **kws
    ) -> None:
        super().__init__(**kws)
        
        self.flow = flow
        self.prob_func = None
        self.trained = False

        self.unnorm_matrix = unnorm_matrix
        if self.unnorm_matrix is None:
            self.unnorm_matrix = torch.eye(ndim)   

        self.train_kws = train_kws
        if self.train_kws is None:
            self.train_kws = {}

        self.train_kws.setdefault("batch_size", 512)
        self.train_kws.setdefault("iters", 1000)
        self.train_kws.setdefault("lr", 0.001)
        self.train_kws.setdefault("lr_min", 0.001)
        self.train_kws.setdefault("lr_decay", 0.99)
        self.train_kws.setdefault("print_freq", 100)
        self.train_kws.setdefault("verbose", 0)

        self.train_history = {}
        self.train_history["loss"] = []
        self.train_history["time"] = []

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z, self.unnorm_matrix.T)
        
    def train(self, prob_func: Callable) -> dict:
        self.prob_func = prob_func
        self.trained = True

        self.train_history = {}
        self.train_history["loss"] = []
        self.train_history["time"] = []

        iters = self.train_kws["iters"]
        batch_size = self.train_kws["batch_size"]
        lr = self.train_kws["lr"]
        lr_min = self.train_kws["lr_min"]
        lr_decay = self.train_kws["lr_decay"]
        print_freq = self.train_kws["print_freq"]
        verbose = self.train_kws["verbose"]
    
        start_time = time.time()

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        for iteration in range(iters):
            x, log_prob = self.flow().rsample_and_log_prob((batch_size,)) 
            x = self.unnormalize(x)

            loss = torch.mean(log_prob) - torch.mean(torch.log(prob_func(x) + 1.00e-15))
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
    
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(lr_min, lr_decay * param_group["lr"])
    
            # Append to history array
            self.train_history["loss"].append(loss.detach())
            self.train_history["time"].append(time.time() - start_time)
    
            # Print update
            if verbose and (iteration % print_freq == 0):
                print(iteration, loss)
        
        return self.train_history

    def __call__(self, prob_func: Callable, size: int) -> torch.Tensor:
        self.trained = False
            
        if not self.trained:
            self.train(prob_func)
        
        with torch.no_grad():
            x = self.flow().sample((size,))
            x = self.unnormalize(x)
            return x