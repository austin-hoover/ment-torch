import torch
import ment_torch as ment
from ment_torch.samp.hmc import compute_gradients
from ment_torch.samp.hmc import compute_gradients_loop


def test_grad_01():

    def func(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x[:, 0]) + torch.cos(x[:, 1])

    def func_grad(x: torch.Tensor) -> torch.Tensor:
        return torch.vstack([torch.cos(x[:, 0]), -torch.sin(x[:, 1])]).T

    x = torch.randn((5, 2))
    g_true = func_grad(x)
    g_calc = compute_gradients(func, x)
    g_calc_loop = compute_gradients_loop(func, x)
    assert torch.all(g_true == g_calc)
    assert torch.all(g_true == g_calc_loop)
    assert torch.all(g_calc == g_calc_loop)


def test_grad_02():

    def func(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x[:, 0]) - 2.0 * (x[:, 0]**2 + x[:, 1]**2 - 2.0) ** 2

    for _ in range(10):
        x = torch.randn((5, 2))
        g_calc = compute_gradients(func, x)
        g_calc_loop = compute_gradients_loop(func, x)
        assert torch.all(g_calc == g_calc_loop)