import itertools
import torch


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