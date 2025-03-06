from typing import Callable
from typing import Union
from typing import Any

import itertools
import torch

from .diag import Histogram
from .prior import InfiniteUniformPrior
from .sim import IdentityTransform
from .sim import LinearTransform


class RegularGridInterpolator:
    # https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py
    def __init__(self, coords: list[torch.Tensor], values: torch.Tensor, fill_value: float = 0.0) -> None:
        self.coords = coords
        self.values = values
        self.fill_value = torch.tensor(fill_value)        

        assert isinstance(self.coords, tuple) or isinstance(self.coords, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.coords)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.coords):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, new_points: torch.Tensor) -> torch.Tensor:
        assert self.coords is not None
        assert self.values is not None

        assert len(new_points) == len(self.coords)
        K = new_points[0].shape[0]
        for x in new_points:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.coords, new_points):
            idx_right = torch.bucketize(x, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.0
            dist_right[dist_right < 0] = 0.0
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.0
 
            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.0
        for indexer in itertools.product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        result = numerator / denominator

        # Handle bounds
        out_of_bounds = torch.zeros(new_points.shape[1], dtype=torch.bool, device=self.values.device)
        for x, c in zip(new_points, self.coords):
            out_of_bounds = out_of_bounds | (x < c[0]) | (x > c[-1])
        result[out_of_bounds] = self.fill_value

        return result

class LagrangeFunction:
    def __init__(
        self, 
        ndim: int, 
        axis: tuple[int] | int, 
        coords: list[torch.Tensor] | torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.ndim = ndim
        self.axis = axis
        
        if type(self.axis) is int:
            self.axis = (self.axis,)
            
        self.coords = coords
        if type(self.coords) is torch.Tensor:
            self.coords = [self.coords]
            
        self.values = self.set_values(values)

        self.limits = [(c[0], c[-1]) for c in self.coords]
        self.limits = torch.tensor(self.limits)

    def set_values(self, values: torch.Tensor) -> None:
        self.values = values
        self.interp = RegularGridInterpolator(self.coords, self.values)
        return self.values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, self.axis]
        return self.interp(x_proj.T)


class MENT:
    def __init__(
        self,
        ndim: int,
        transforms: list[Callable],
        projections: list[list[Histogram]],
        prior: Any,
        sampler: Callable,
        unnorm_matrix: torch.Tensor = None,
        nsamp: int = 1_000_000,
        integration_limits: list[tuple[float, float]] = None,
        integration_size: int = None,
        store_integration_points: bool = True,
        verbose: int = 1,
        mode: str = "sample",
    ) -> None:
        """Constructor."""
        self.ndim = ndim
        self.verbose = int(verbose)
        self.mode = mode

        self.transforms = transforms
        self.projections = self.set_projections(projections)

        self.diagnostics = []
        for index in range(len(self.projections)):
            self.diagnostics.append([hist.copy() for hist in self.projections[index]])

        self.prior = prior
        if self.prior is None:
            self.prior = InfiniteUniformPrior(ndim=ndim)

        self.unnorm_matrix = unnorm_matrix
        self.unnorm_transform = self.set_unnorm_transform(unnorm_matrix)

        self.lagrange_functions = self.init_lagrange_functions()

        self.sampler = sampler
        self.nsamp = int(nsamp)

        self.integration_limits = integration_limits
        self.integration_size = integration_size
        self.integration_points = None
        self.store_integration_points = store_integration_points

        self.epoch = 0

    def set_unnorm_transform(self, unnorm_matrix: torch.Tensor) -> Callable:
        """Set inverse of normalization matrix.

        The unnormalization matrix transforms normalized coordinates z to
        phase space coordinates x via the linear mapping: x = Vz.
        """
        self.unnorm_matrix = unnorm_matrix
        if self.unnorm_matrix is None:
            self.unnorm_transform = IdentityTransform()
            self.unnorm_matrix = torch.eye(self.ndim)
        else:
            self.unnorm_transform = LinearTransform(self.unnorm_matrix)
        return self.unnorm_transform

    def set_projections(self, projections: list[list[Histogram]]) -> list[list[Histogram]]:
        """Set list of measured projections (histograms)."""
        self.projections = projections
        if self.projections is None:
            self.projections = [[]]
        return self.projections

    def init_lagrange_functions(self, **interp_kws) -> list[list[LagrangeFunction]]:
        """Initialize lagrange multipler functions.

        The function h(u_proj) = 1 if the measured projection g(u_proj) > 0,
        otherwise h(u_proj) = 0.

        Key word arguments passed to `LagrangeFunction` constructor.
        """
        self.lagrange_functions = []
        for index in range(len(self.projections)):
            self.lagrange_functions.append([])
            for projection in self.projections[index]:
                values = torch.zeros(projection.shape)
                values[projection.values > 0.0] = 1.0
                lagrange_function = LagrangeFunction(
                    ndim=projection.ndim,
                    axis=projection.axis,
                    coords=projection.coords,
                    values=values,
                )
                self.lagrange_functions[-1].append(lagrange_function)
        return self.lagrange_functions

    def unnormalize(self, z: torch.Tensor) -> torch.Tensor:
        """Unnormalize coordinates z: x = Vz."""
        if self.unnorm_transform is None:
            self.unnorm_transform = IdentityTransform()
        return self.unnorm_transform(z)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates x: z = V^-1 z."""
        return self.unnorm_transform.inverse(x)




        