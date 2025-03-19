import argparse
import math

import matplotlib.pyplot as plt
import torch
import zuko

import ment_torch as ment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamp", type=int, default=10_000)
    parser.add_argument("--chains", type=int, default=1)
    return parser.parse_args()


def main(args):

    def prob_func(x: torch.Tensor) -> torch.Tensor:
        log_prob = torch.sin(torch.pi * x[:, 0]) - 2.0 * (x[:, 0]**2 + x[:, 1]**2 - 2.0) ** 2
        return torch.exp(log_prob)

    x_start = torch.randn((args.chains, 2))
    x_start = x_start * 0.25**2

    sampler = ment.HamiltonianMonteCarloSampler(
        ndim=2,
        start=x_start,
        step_size=0.25,
        steps_per_samp=3,
        verbose=1,
    )
    x_pred = sampler(prob_func=prob_func, size=args.nsamp)
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
    grid_values = prob_func(grid_points).reshape(grid_shape)

    fig, axs = plt.subplots(ncols=2, figsize=(6.0, 2.75), sharex=True, sharey=True)
    axs[0].hist2d(x_pred[:, 0], x_pred[:, 1], bins=80, range=grid_limits)
    axs[1].pcolormesh(grid_coords[0], grid_coords[1], grid_values.T)
    axs[0].set_title("PRED", fontsize="medium")
    axs[1].set_title("TRUE", fontsize="medium")
    plt.show()


if __name__ == "__main__":
    main(parse_args())