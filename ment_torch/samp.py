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
    idx = torch.multinomial(pdf, num_samples=size, replacement=True, generator=rng)
    return items[idx]


def random_shuffle(items: torch.Tensor, rng: torch.Generator = None) -> torch.Tensor:
    idx = torch.randperm(items.shape[0])
    return items[idx]


class Sampler:
    """Base class for particle samplers."""
    def __init__(
        self,
        ndim: int,
        verbose: int = 0, 
        device: torch.device = None, 
        seed: int = None,
        noise: float = 0.0,
        noise_type: float = "gaussian",
    ) -> None:
        self.ndim = ndim
        self.verbose = verbose
        self.device = device
        
        self.seed = seed
        self.rng = torch.Generator()
        if self.seed is not None:
            self.rng.manual_seed(self.seed)
            
        self.noise = noise
        self.noise_type = noise_type

        self.results = {}

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        x_add = torch.zeros(x.shape, device=self.device)
        if self.noise_type == "uniform":
            x_add = random_uniform(-0.5, 0.5, device=device, rng=self.rng)
            x_add = x_add * self.noise_scale
        elif self.noise_type == "gaussian":
            x_add = torch.randn(x.shape)
            x_add = x_add * self.noise_scale
        return x + x_add  

    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        raise NotImplementedError    

    def __call__(self, prob_func: Callable, size: int) -> torch.Tensor:
        self.results = {}
        x = self._sample(prob_func, size)
        if self.noise:
            x = self.add_noise(x)
        return x


class GridSampler(Sampler):
    """Samples from discretized distribution on regular grid."""
    def __init__(
        self,
        limits: list[tuple[float]],
        shape: tuple[int],
        noise: float = 0.0,
        store: bool = True,
        **kws
    ) -> None:
        super().__init__(self, **kws)
        
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

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def _sample(self, prob_func: Callable, size: int) -> torch.Tensor:
        values = prob_func(self.get_grid_points())    
        values = values / torch.sum(values)
        
        idx = torch.squeeze(torch.nonzero(values))
        idx = random_choice(idx, size, pdf=values[idx])
        idx = torch.unravel_index(idx, self.shape)
    
        x = torch.zeros((size, self.ndim), device=self.device)
        for axis in range(self.ndim):
            lb = self.edges[axis][idx[axis]]
            ub = self.edges[axis][idx[axis] + 1]
            x[:, axis] = random_uniform(lb, ub, size, device=self.device, rng=self.rng)

            if self.noise:
                delta = (ub - lb) * self.noise
                x[:, axis] += 0.5 * random_uniform(-delta, delta, size, device=self.device, rng=self.rng)

        return torch.squeeze(x)
        

class MetropolisHastingsSampler(Sampler):
    """Samples using Metropolis-Hastings algorithm.

    https://colindcarroll.com/2019/08/18/very-parallel-mcmc-sampling/
    """
    def __init__(
        self,
        chains: int = 1,
        burnin: int = 0,
        start: torch.Tensor = None,
        proposal_cov: np.ndarray = None,
        shuffle: bool = False,
        **kws
    ) -> None:
        """Constructor.
        
        Parameters
        ----------
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
        """
        super().__init__(**kws)
        
        self.chains = chains
        self.burnin = burnin
        self.start = start

        self.proposal_mean = torch.zeros(self.ndim, device=self.device)
        self.proposal_cov = proposal_cov
        if self.proposal_cov is None:
            self.proposal_cov = torch.eye(self.ndim, device=self.device)
        self.proposal_dist = torch.distributions.MultivariateNormal(self.proposal_mean, self.proposal_cov)
        
        self.shuffle = shuffle
                
    def _sample(self, prob_func: Callable, size: int) -> np.ndarray:
        # Initialize list of points. From now on we each "point" is really a batch of 
        # size (nchains, ndim). Burnin-points will be discarded later.
        size = int(math.ceil(size / float(self.chains)))
        size = size + self.burnin
        points = torch.zeros((size, self.chains, self.ndim), device=self.device) 
    
        # Sample proposal points from a Gaussian distribution. (The means will be updated
        # during the random walk.)
        proposal_points = self.proposal_dist.sample((size - 1, self.chains))
    
        # Set starting point for each chain. If none is provided, sample from the proposal
        # distribution centered at the origin.
        start = self.start
        if start is None:
            start = self.proposal_dist.sample((self.chains,))
            start *= 0.50
        if self.chains == 1:
            start = torch.zeros((self.chains, self.ndim))
        points[0] = start
    
        # Execute random walk
        random_uniforms = random_uniform(
            lb=0.0, 
            ub=1.0, 
            size=(size - 1, self.chains), 
            rng=self.rng, 
            device=self.device
        )
        accept = torch.zeros(self.chains, device=self.device)

        points[0] = start
        prob = prob_func(start)
    
        self.results = {}
        self.results["n_total_accepted"] = 0
        self.results["n_total"] = 0
        self.results["acceptance_rate"] = None
    
        for i in tqdm_wrapper(range(1, size), self.verbose):
            proposal_point = points[i - 1] + proposal_points[i - 1]
            proposal_prob = prob_func(proposal_point)
            accept = proposal_prob > prob * random_uniforms[i - 1]
    
            if i > self.burnin:
                self.results["n_total_accepted"] += torch.count_nonzero(accept)
                self.results["n_total"] += self.chains
                self.results["acceptance_rate"] = self.results["n_total_accepted"] / self.results["n_total"]
                if self.verbose > 2:
                    print(f"debug {i:05.0f}")
                    print(self.results)
    
            points[i] = points[i - 1]
            points[i][accept] = proposal_point[accept]
            prob[accept] = proposal_prob[accept]
    
        points = points[self.burnin:]

        # Debug
        if self.verbose > 1:        
            print("debug acceptance rate =", self.results["acceptance_rate"])
            for axis in range(self.ndim):
                x_chain_stds = [torch.std( x_chain[:, axis]) for x_chain in points]
                x_chain_avgs = [torch.mean(x_chain[:, axis]) for x_chain in points]
                print(f"debug axis={axis} between-chain avg(x_chain_std) =", torch.mean(x_chain_stds))
                print(f"debug axis={axis} between-chain std(x_chain_std) =", torch.std( x_chain_stds))
                print(f"debug axis={axis} between-chain avg(x_chain_avg) =", torch.mean(x_chain_avgs))
                print(f"debug axis={axis} between-chain std(x_chain_avg) =", torch.std( x_chain_avgs))
        
                x_avg = torch.mean(torch.hstack([chain[:, axis] for chain in points]))
                x_std = torch.std( torch.hstack([chain[:, axis] for chain in points]))
                print(f"debug axis={axis} x_std =", x_std)
                print(f"debug axis={axis} x_avg =", x_avg)

        # Merge chains
        points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])

        # Shuffle
        if self.shuffle:
            points = random_shuffle(points, rng=self.rng)

        # Make sure size is correct
        size = size * self.chains
        points = points[:size]
        return points


class FlowSampler(Sampler):
    """Samples using normalizing flow trained by reverse KLD."""
    def __init__(
        self, flow: zuko.flows.Flow, unnorm_matrix: torch.Tensor = None, train_kws: dict = None, **kws
    ) -> None:
        super().__init__(**kws)
        
        self.flow = flow
        self.prob_func = None
        self.trained = False

        self.unnorm_matrix = unnorm_matrix
        if self.unnorm_matrix is None:
            self.unnorm_matrix = torch.eye(self.ndim)   

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

        self.results = {}
        self.results["loss"] = []
        self.results["time"] = []

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z, self.unnorm_matrix.T)
        
    def train(self, prob_func: Callable) -> dict:
        self.prob_func = prob_func
        self.trained = True

        self.results = {}
        self.results["loss"] = []
        self.results["time"] = []

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
            self.results["loss"].append(loss.detach())
            self.results["time"].append(time.time() - start_time)
    
            # Print update
            if verbose and (iteration % print_freq == 0):
                print(iteration, loss)
        
    def __call__(self, prob_func: Callable, size: int) -> torch.Tensor:
        self.trained = False
            
        if not self.trained:
            self.train(prob_func)
        
        with torch.no_grad():
            x = self.flow().sample((size,))
            x = self.unnormalize(x)
            return x