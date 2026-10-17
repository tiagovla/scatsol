"""TM scattering from a PEC cylinder with a dielectric coating."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from scatsol import (
    CylindricalGeometry,
    CylindricalRegion,
    CylindricalSolver,
    Material,
    Polarization,
)


def solve_pec_cylinder_coated(polarization=Polarization.TM):
    """Return the coated PEC geometry and scattering solution for one polarization."""
    pec = Material.PEC()
    dielectric = Material(epsilon_r=2.5)
    vacuum = Material.vacuum()
    geometry = CylindricalGeometry(
        [
            CylindricalRegion(0.0, 0.5, pec),
            CylindricalRegion(0.5, 0.8, dielectric),
            CylindricalRegion(0.8, float("inf"), vacuum),
        ]
    )
    solution = CylindricalSolver(geometry, frequency=300e6).solve(n=100, polarization=polarization)
    return geometry, solution


def plot_pec_cylinder_coated(geometry, solution, extent=3.0, resolution=200):
    """Plot the real longitudinal field component for the solution's polarization."""
    axis = np.linspace(-extent, extent, resolution)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    if solution.polarization == Polarization.TM:
        field = solution.e_z(points).real
        component = r"$E_z$"
    else:
        field = solution.h_z(points).real
        component = r"$H_z$"

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))
    image = ax.tripcolor(points[:, 0], points[:, 1], field, shading="gouraud", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax)
    for region in geometry.regions[:-1]:
        ax.add_patch(Circle((0, 0), region.outer_radius, fill=False, color="black"))
    ax.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"{solution.polarization.value} scattering by a coated PEC cylinder - {component}",
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        aspect="equal",
    )
    return fig, ax


def main():
    for polarization in (Polarization.TM, Polarization.TE):
        geometry, solution = solve_pec_cylinder_coated(polarization)
        plot_pec_cylinder_coated(geometry, solution)
    plt.show()


if __name__ == "__main__":
    main()
