import itertools
import torch


def unravel(iterable):
    return list(itertools.chain.from_iterable(iterable))


def rotation_matrix(angle: float) -> torch.Tensor:
    angle = torch.tensor(angle)
    return torch.tensor([[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]])