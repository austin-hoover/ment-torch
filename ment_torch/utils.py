import itertools
import torch
from tqdm import tqdm


def unravel(iterable):
    return list(itertools.chain.from_iterable(iterable))


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


def wrap_tqdm(iterable, verbose=True):
    return tqdm(iterable) if verbose else iterable


def random_choice(items: torch.Tensor, size: int, pdf: torch.Tensor, rng: torch.Generator = None) -> torch.Tensor:
    idx = torch.multinomial(pdf, num_samples=size, replacement=True, generator=rng)
    return items[idx]
    

def random_shuffle(items: torch.Tensor, rng: torch.Generator = None) -> torch.Tensor:
    idx = torch.randperm(items.shape[0])
    return items[idx]


def random_uniform(
    lb: torch.Tensor | float,
    ub: torch.Tensor | float,
    size: int,
    rng: torch.Generator = None,
    device: torch.device = None,
) -> torch.Tensor:
    return lb + (ub - lb) * torch.rand(size, device=device, generator=rng)


def rotation_matrix(angle: float) -> torch.Tensor:
    angle = torch.tensor(float(angle))
    return torch.tensor(
        [[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]]
    )

