import itertools
import torch


def unravel(iterable):
    return list(itertools.chain.from_iterable(iterable))


def rotation_matrix(angle: float) -> torch.Tensor:
    angle = torch.tensor(float(angle))
    return torch.tensor([[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]])


def coords_to_edges(coords: torch.Tensor) -> torch.Tensor:
    delta = coords[1] - coords[0]
    edges = torch.zeros(len(coords) + 1)
    edges[:-1] = coords - 0.5 * delta
    edges[-1] = coords[-1] + delta
    return edges


def edges_to_coords(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])


def get_grid_points(grid_coords: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([c.ravel() for c in torch.meshgrid(*grid_coords, indexing="ij")], axis=-1)