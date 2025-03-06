import math
import time
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import torch
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


class GridSampler:
    def __init__(
        self,
        limits: list[tuple[float]],
        shape: tuple[int],
        noise: float = 0.0,
        store: bool = True,
        device=None,
    ) -> None:
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