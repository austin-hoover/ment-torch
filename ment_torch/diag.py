import copy
import math
from typing import Union
from typing import Self

import torch


def coords_to_edges(coords: torch.Tensor) -> torch.Tensor:
    delta = coords[1] - coords[0]
    edges = torch.zeros(len(coords) + 1)
    edges[:-1] = coords - 0.5 * delta
    edges[-1] = coords[-1] + delta
    return edges


def edges_to_coords(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])


def get_grid_points(grid_coords) -> torch.Tensor:
    return torch.stack([C.ravel() for C in torch.meshgrid(*grid_coords, indexing="ij")], axis=-1)


class Histogram:
    def __init__(self) -> None:
        return
        

class HistogramND(Histogram):
    def __init__(
        self,
        axis: tuple[int, ...],
        edges: list[torch.Tensor] = None,
        coords: list[torch.Tensor] = None,
        values: torch.Tensor = None,
    ) -> None:
        super().__init__()
        
        self.axis = axis
        self.ndim = len(axis)

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = [edges_to_coords(e) for e in self.edges]
        if self.edges is None and self.coords is not None:
            self.edges = [coords_to_edges(c) for c in self.coords]

        self.shape = tuple([len(c) for c in self.coords])
        self.values = values
        if self.values is None:
            self.values = torch.zeros(self.shape)

        self.bin_sizes = [self.coords[i][1] - self.coords[i][0] for i in range(self.ndim)]
        self.bin_volume = torch.prod(self.bin_sizes)
        self.grid_shape = tuple([len(c) for c in self.coords])

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def normalize(self) -> None:
        values = torch.clone(self.values)
        values_sum = torch.sum(values)
        if values_sum > 0.0:
            values = values / values_sum / self.bin_volume
        self.values = values

    def get_grid_points(self) -> torch.Tensor:
        return get_grid_points(self.coords)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.axis]

    def bin(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.project(x)

        hist_obj = torch.histogramdd(x_proj, bins=self.edges, density=True)
        values = hist_obj.hist
        
        self.values = values
        self.normalize()
        return self.values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.bin(x)


class Histogram1D(Histogram):
    def __init__(
        self,
        axis: int = 0,
        edges: torch.Tensor = None,
        coords: torch.Tensor = None,
        values: torch.Tensor = None,
    ) -> None:
        super().__init__()

        self.axis = axis
        self.ndim = 1

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = edges_to_coords(self.edges)
        if self.edges is None and self.coords is not None:
            self.edges = coords_to_edges(self.coords)

        self.shape = len(self.coords)
        self.values = values
        if self.values is None:
            self.values = torch.zeros(self.shape)

        self.bin_size = self.coords[1] - self.coords[0]
        self.bin_volume = self.bin_width = self.bin_size
        
    def copy(self) -> Self:
        return copy.deepcopy(self)

    def get_grid_points(self) -> torch.Tensor:
        return self.coords

    def normalize(self) -> None:
        values = torch.clone(self.values)
        values_sum = torch.sum(values)
        if values_sum > 0.0:
            values = values / values_sum / self.bin_volume
        self.values = values

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.axis]

    def bin(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.project(x)

        hist_obj = torch.histogram(x_proj, bins=self.edges, density=True)
        values = hist_obj.hist
        
        self.values = values
        self.normalize()
        return self.values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.bin(x)
