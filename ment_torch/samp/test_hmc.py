import argparse
import math

import matplotlib.pyplot as plt
import torch
import zuko

import ment_torch as ment
from ment_torch.samp.hmc import sample


parser = argparse.ArgumentParser()
parser.add_argument("--nsamp", type=int, default=10_000)
parser.add_argument("--chains", type=int, default=1)
args = parser.parse_args()


# Define PDF
def log_prob_func(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x[:, 0]) - 2.0 * (x[:, 0]**2 + x[:, 1]**2 - 2.0) ** 2

# Sample
x_start = torch.zeros((args.chains, 2))
x_pred = sample(
    log_prob_func=log_prob_func,
    start=x_start,
    size=args.nsamp,
    step_size=0.25,
    steps_per_samp=3,
    debug=True,
)
x_pred = torch.vstack(x_pred)
print(x_pred.shape)


# Plot
xmax = 3.0
grid_limits = 2 * [(-xmax, xmax)]
grid_shape = (128, 128)
grid_coords = [
    torch.linspace(grid_limits[i][0], grid_limits[i][1], grid_shape[i])
    for i in range(2)
]
grid_points = torch.vstack([c.ravel() for c in torch.meshgrid(*grid_coords, indexing="ij")]).T
grid_values = torch.exp(log_prob_func(grid_points))
grid_values = grid_values.reshape(grid_shape)

fig, axs = plt.subplots(ncols=2, figsize=(6.0, 2.75), sharex=True, sharey=True)
axs[0].hist2d(x_pred[:, 0], x_pred[:, 1], bins=80, range=grid_limits)
axs[1].pcolormesh(grid_coords[0], grid_coords[1], grid_values.T)
axs[0].set_title("PRED", fontsize="medium")
axs[1].set_title("TRUE", fontsize="medium")
plt.show()
