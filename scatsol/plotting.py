from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scatsol.geometry import CylindricalGeometry
import numpy as np


def plot_geometry(geometry: CylindricalGeometry, show: bool = False) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(constrained_layout=True)
    phi = np.linspace(0, 2 * np.pi, 100)
    for region in geometry.regions:
        ax.plot(region.inner_radius * np.cos(phi), region.inner_radius * np.sin(phi), color="black")
        ax.text(
            region.inner_radius * np.cos(np.pi / 4),
            region.inner_radius * np.sin(np.pi / 4),
            rf"$\mu_r$={region.material.mu_r}$, \epsilon_r$={region.material.epsilon_r}",
        )
    ax.set_aspect("equal")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Cylindrical Geometry")
    if show:
        plt.show()
    return fig, ax
