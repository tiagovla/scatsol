from scatsol import __version__
from scatsol.constant import Polarization
from scatsol.geometry import CylindricalGeometry
from scatsol.material import Material
from scatsol.plotting import plot_geometry
from scatsol.region import CylindricalRegion
from scatsol.solver import CylindricalSolver
import numpy as np
import scatsol

import matplotlib.pyplot as plt


def test_version():
    assert __version__ == "0.0.2"


def test_problem():
    free_space_m = Material(epsilon_r=1.0, mu_r=1.0)
    cylinder_m = Material(epsilon_r=4.0, mu_r=1.0)
    coating_m = Material(epsilon_r=3.0, mu_r=1.0)

    cylinder_r = CylindricalRegion(inner_radius=0.0, outer_radius=0.5, material=cylinder_m)
    coating_r = CylindricalRegion(inner_radius=0.5, outer_radius=1.0, material=coating_m)
    free_space_r = CylindricalRegion(
        inner_radius=1.0, outer_radius=scatsol.INFTY, material=free_space_m
    )

    geometry = CylindricalGeometry([cylinder_r, coating_r, free_space_r])
    # plot_geometry(geometry=geometry, show=False)

    solver = CylindricalSolver(geometry=geometry, frequency=300e6)
    solution = solver.solve(polarization=Polarization.TM, n=50)
    x, y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    xyz = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    e_inc_field, _ = solution.incident_field(xyz)

    fig, ax = plt.subplots()
    ax.tripcolor(*xyz[:, np.array([0, 1])].T, e_inc_field[:, 2].real)
    print(e_inc_field[:, 2].real)
    plt.show()

    assert False
