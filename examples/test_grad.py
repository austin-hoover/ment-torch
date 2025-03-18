from typing import Callable
import math
import matplotlib.pyplot as plt
import torch


def compute_gradients_loop(func: Callable[torch.Tensor, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x.requires_grad_(True)
    f = func(x)
    for i in range(x.shape[0]):
        f[i].backward(retain_graph=True)
    x.requires_grad_(False)
    return x.grad

def compute_gradients(func: Callable[torch.Tensor, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x.requires_grad_(True)
    f = func(x)
    f.sum().backward()
    x.requires_grad_(False)
    return x.grad


# Test
def func(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x[:, 0]) + torch.cos(x[:, 1])

def func_grad(x: torch.Tensor) -> torch.Tensor:
    return torch.vstack([torch.cos(x[:, 0]), -torch.sin(x[:, 1])]).T

x = torch.randn((5, 2))
f = func(x)
g = compute_gradients(func, x)
g_exp = func_grad(x)

print("x:")
print(x)
print("f(x):")
print(f)
print("g(x) - g(x) expected:")
print(g - g_exp)