from scatsol import __version__
from scatsol import (
    Polarization,
    CylindricalGeometry,
    Material,
    CylindricalRegion,
    CylindricalSolver,
)
import numpy as np
import scatsol

import matplotlib.pyplot as plt

from scatsol.plotting import plot_geometry


def test_version():
    assert __version__ == "0.0.2"


def test_problem():
    diel = Material(epsilon_r=4.00, mu_r=1.0)
    free_space = Material.free_space()

    cylinder_r = CylindricalRegion(inner_radius=0.0, outer_radius=0.5, material=diel)
    coating_r = CylindricalRegion(inner_radius=0.5, outer_radius=1.0, material=diel)
    free_space_r = CylindricalRegion(1.0, float("inf"), material=free_space)

    geometry = CylindricalGeometry([cylinder_r, coating_r, free_space_r])
    # plot_geometry(geometry=geometry)

    solver = CylindricalSolver(geometry=geometry, frequency=300e6)
    solution = solver.solve(polarization=Polarization.TM, n=50)

    x, y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    xyz = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    e_field, _ = solution.total_field(xyz)

    _, ax = plt.subplots(constrained_layout=True)
    ax.tripcolor(*xyz[:, [0, 1]].T, e_field[:, 2].real, shading="gouraud", vmin=-1, vmax=1)
    for r in geometry.regions[:-1]:
        phi = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r.outer_radius * np.cos(phi), r.outer_radius * np.sin(phi), "k")
    ax.margins(0)
    plt.show()

    assert False
