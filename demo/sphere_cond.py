import matplotlib.pyplot as plt
import numpy as np
import scatsol.sphere

from scatsol import Material


def main() -> None:
    radius = 1.0
    frequency = 300e6  # 300MHz
    vacuum = Material(epsilon_r=1, mu_r=1)

    interval = np.linspace(-5, 5, 50)
    x, z = np.meshgrid(interval, interval)
    xyz = np.stack((x.ravel(), 0 * x.ravel(), z.ravel()), axis=1)

    _, Hrtp = scatsol.sphere.mie_spherical_scattered_field(
        xyz, radius, frequency, vacuum, n=50
    )

    fig, ax = plt.subplots(constrained_layout=True)
    h = ax.tripcolor(
        *xyz[:, np.array([0, 2])].T,
        vacuum.eta * Hrtp[:, 2].real,
        shading="gouraud",
        vmin=-1,
        vmax=1,
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
    ax.set_title(r"$\eta_0 H_\phi$")


if __name__ == "__main__":
    main()
    plt.show()
