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


def random_uniform(lb: float, ub: float, size: int, device=None) -> torch.Tensor:
    return lb + (ub - lb) * torch.rand(size, device=device)


def random_choice(items: torch.tensor, size: int, p: torch.Tensor):
    return items[p.multinomial(num_samples=size, replacement=True)]


def sample_hist_bins(values: torch.Tensor, size: int) -> torch.Tensor:
    pdf = torch.ravel(values) + 1.00e-15
    idx = torch.squeeze(torch.nonzero(pdf))
    idx = random_choice(idx, size, p=(pdf / torch.sum(pdf)))
    return idx


def sample_hist(
    values: torch.Tensor,
    edges: List[torch.Tensor],
    size: int,
    noise: float = 0.0,
    device: torch.device = None,
) -> torch.Tensor:
    ndim = values.ndim
    if ndim == 1:
        edges = [edges]

    idx = sample_hist_bins(values, size)
    idx = torch.unravel_index(idx, values.shape)

    x = torch.zeros((size, ndim), device=device)
    for axis in range(ndim):
        lb = edges[axis][idx[axis]]
        ub = edges[axis][idx[axis] + 1]
        x[:, axis] = random_uniform(lb, ub, size, device=device)
        if noise:
            delta = ub - lb
            x[:, axis] += 0.5 * random_uniform(-delta, delta, size, device=device)
    x = torch.squeeze(x)
    return x


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
        if not (prob_func is self.prob_func):
            self.trained = False
            
        if not self.trained:
            self.train(prob_func)
        
        with torch.no_grad():
            x = self.flow().sample((size,))
            x = self.unnormalize(x)
            return x