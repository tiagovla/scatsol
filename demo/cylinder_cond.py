import matplotlib.pyplot as plt
import numpy as np

import scatsol


def main() -> None:
    radius = 1.0
    frequency = 300e6  # 300MHz
    vacuum = scatsol.Material(epsilon_r=1.0, mu_r=1.0)

    interval = np.linspace(-5.0, 5.0, 100)
    x, y = np.meshgrid(interval, interval)
    xyz = np.stack((x.ravel(), y.ravel(), 0 * x.ravel()), axis=1)

    E, _ = scatsol.cylinder.mie_total_field(xyz, radius, frequency, vacuum, n=100)
    _, H = scatsol.cylinder.mie_total_field(xyz, radius, frequency, vacuum, n=100, pol="TE")

    fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2.5))

    h0 = ax0.tripcolor(*xyz[:, np.array([0, 1])].T, E[:, 2].real, shading="gouraud", vmin=-2, vmax=2, rasterized=True)
    h1 = ax1.tripcolor(*xyz[:, np.array([0, 1])].T, H[:, 2].real, shading="gouraud", vmin=-2, vmax=2, rasterized=True)
    cb0 = fig.colorbar(h0, ax=ax0)
    cb1 = fig.colorbar(h1, ax=ax1)
    for ax in (ax0, ax1):
        ax.plot(
            radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
            radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
            lw=1.0,
            color="white",
        )
        ax.set(xlabel="$x$ [$\lambda_0$]", ylabel="$y$ [$\lambda_0$]")
        ax.margins(0)
    ax0.set_title("(a)")
    ax1.set_title("(b)")
    cb0.ax.set_title(r"$E_z$ [V/m]")
    cb1.ax.set_title(r"$H_z$ [A/m]")
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")

    fig.savefig("figures/solution_cylinder_cond.pgf", dpi=300)


if __name__ == "__main__":
    main()
    plt.show()
