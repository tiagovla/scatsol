import matplotlib.pyplot as plt
import numpy as np

from scatsol import Material
import scatsol.sphere
import scatsol.utils


def main() -> None:
    radius = 1.0  # 1m
    frequency = 300e6  # 300MHz
    vacuum = Material(epsilon_r=1, mu_r=1)
    sphere = Material(epsilon_r=2.56, mu_r=1)

    interval = np.linspace(-3, 3, 80)
    x, z = np.meshgrid(interval, interval)
    xyz = np.stack((x.ravel(), 0 * x.ravel(), z.ravel()), axis=1)

    _, H = scatsol.sphere.mie_total_field(xyz, radius, frequency, background=vacuum, sphere=sphere, n=50)
    H = scatsol.utils.field_sph2cart(H, xyz)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))
    h = ax.tripcolor(
        *xyz[:, np.array([0, 2])].T,
        vacuum.eta * H[:, 1].real,
        shading="gouraud",
        vmin=-2,
        vmax=2,
    )
    fig.colorbar(h, ax=ax)
    ax.plot(
        radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
        radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
        lw=1.0,
        color="white",
    )
    ax.set(xlabel="$x$", ylabel="$z$")
    ax.margins(0)
    ax.set_title(r"$\eta_0 H_y$")


if __name__ == "__main__":
    main()
    plt.show()
