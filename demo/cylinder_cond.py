import matplotlib.pyplot as plt
import numpy as np

from scatsol import Material
import scatsol.cylinder


def main() -> None:
    radius = 1.0
    frequency = 300e6  # 300MHz
    vacuum = Material(epsilon_r=1.0, mu_r=1.0)

    interval = np.linspace(-5.0, 5.0, 100)
    x, y = np.meshgrid(interval, interval)
    xyz = np.stack((x.ravel(), y.ravel(), 0 * x.ravel()), axis=1)

    E, _ = scatsol.cylinder.mie_total_field(xyz, radius, frequency, vacuum, n=50)
    _, H = scatsol.cylinder.mie_total_field(xyz, radius, frequency, vacuum, n=50, pol="TE")

    fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))

    h0 = ax0.tripcolor(*xyz[:, np.array([0, 1])].T, E[:, 2].real, shading="gouraud", vmin=-2, vmax=2)
    h1 = ax1.tripcolor(*xyz[:, np.array([0, 1])].T, H[:, 2].real, shading="gouraud", vmin=-2, vmax=2)
    fig.colorbar(h0, ax=ax0)
    fig.colorbar(h1, ax=ax1)
    for ax in (ax0, ax1):
        ax.plot(
            radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
            radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
            lw=1.0,
            color="white",
        )
        ax.set(xlabel="$x$", ylabel="$y$")
        ax.margins(0)
    ax0.set_title(r"$E_z$ - TMz")
    ax1.set_title(r"$H_z$ - TEz")


if __name__ == "__main__":
    main()
    plt.show()
