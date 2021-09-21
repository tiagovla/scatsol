from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from scatsol.pec_cylinder_2d import SolutionTM
from scatsol.utils import generate_position


def plot_field(F: np.ndarray, x: np.ndarray, func: Callable, title: str = ""):
    directions = ["x-dir", "y-dir", "z-dir"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].tripcolor(x[:, 0], x[:, 1], func(F[:, 0]))
    axs[1].tripcolor(x[:, 0], x[:, 1], func(F[:, 1]))
    axs[2].tripcolor(x[:, 0], x[:, 1], func(F[:, 2]))

    for ax, dir in zip(axs, directions):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(" ".join([title, dir]))

    fig.tight_layout()


def main():

    k_0 = 2 * np.pi  # -> lambda = 1 [m]
    eta_0 = 120 * np.pi
    radius = 0.5

    x = generate_position(
        np.linspace(-2, 2, 50), np.linspace(-2, 2, 50), np.linspace(0, 0, 1)
    )

    sol_tm = SolutionTM(radius=radius, k=k_0, eta=eta_0)
    E_inc = sol_tm.E_inc(x)
    H_inc = sol_tm.H_inc(x)
    E_scat = sol_tm.E_scat(x)
    H_scat = sol_tm.H_scat(x)

    plot_field(E_inc, x, np.real, title="Real $E_{inc}$")
    plot_field(H_inc, x, np.real, title="Real $H_{inc}$")
    plot_field(E_scat, x, np.real, title="Real $E_{scat}$")
    plot_field(H_scat, x, np.real, title="Real $H_{scat}$")

    plt.show()


if __name__ == "__main__":
    main()
