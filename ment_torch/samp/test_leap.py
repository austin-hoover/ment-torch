import matplotlib.pyplot as plt
import numpy as np
import torch
from ment_torch.samp.hmc import leapfrog


def log_prob_func(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x[:, 0]) - 2.0 * (x[:, 0]**2 + x[:, 1]**2 - 2.0) ** 2


ndim = 2
nsamp = 4

x = torch.randn((nsamp, ndim))
p = torch.zeros((nsamp, ndim))

xs, ps = leapfrog(x, p, log_prob_func)

xs = torch.stack(xs)
ps = torch.stack(ps)

fig, axs = plt.subplots(ncols=2)
index = 0
axs[0].scatter(xs[:, index, 0], xs[:, index, 1])
axs[1].scatter(ps[:, index, 0], ps[:, index, 1])
plt.show()