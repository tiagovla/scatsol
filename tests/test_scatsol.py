from scatsol import __version__
from scatsol.material import Material
from scatsol.region import CylindricalRegion
from scatsol.plotting import plot_geometry
import numpy as np

from scatsol.solver import CylindricalSolver, CylindricalGeometry
from scatsol.constant import Polarization


def test_version():
    assert __version__ == "0.1.0"


def test_problem():
    free_space_material = Material(epsilon_r=1.0, mu_r=1.0)
    cylinder_material = Material(epsilon_r=4.0, mu_r=1.0)
    coating_material = Material(epsilon_r=3.0, mu_r=1.0)

    cylinder = CylindricalRegion(inner_radius=0.0, outer_radius=0.5, material=cylinder_material)
    coating = CylindricalRegion(inner_radius=0.5, outer_radius=1.0, material=coating_material)
    free_space = CylindricalRegion(inner_radius=1.0, outer_radius=float("inf"), material=free_space_material)

    geometry = CylindricalGeometry([cylinder, coating, free_space])
    plot_geometry(geometry=geometry)

    solver = CylindricalSolver(geometry=geometry, frequency=300e6)
    solution = solver.solve(polarization=Polarization.TM, n=10)
    e_inc_field, h_inc_field = solution.incident_field([0.0, 0.0, 0.0])
    e_scat_field, h_scat_field = solution.scattered_field([0.0, 0.0, 0.0])

    assert False
