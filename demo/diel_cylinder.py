"""Solve and plot scattering from a dielectric cylinder."""

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


def solve_diel_cylinder():
    """Return the geometry and solution for the dielectric-cylinder example."""
    dielectric = Material(epsilon_r=2.0)
    vacuum = Material.vacuum()
    geometry = CylindricalGeometry(
        [
            CylindricalRegion(0.0, 0.5, dielectric),
            CylindricalRegion(0.5, 0.8, vacuum),
            CylindricalRegion(0.8, float("inf"), dielectric),
        ]
    )
    solution = CylindricalSolver(geometry, frequency=300e6).solve(
        polarization=Polarization.TM, n=100
    )
    return geometry, solution


def plot_diel_cylinder(geometry, solution, extent=3.0, resolution=100):
    """Plot the real z component of the electric field and return the figure and axes."""
    axis = np.linspace(-extent, extent, resolution)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    e_z = solution.e_z(points).real
    xy = points[:, :2].T

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))
    image = ax.tripcolor(*xy, e_z, shading="gouraud", vmin=-1, vmax=1)
    for region in geometry.regions[:-1]:
        ax.add_patch(Circle((0, 0), region.outer_radius, fill=False, color="black"))
    ax.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title=r"$E_z$ field for a layered dielectric cylinder",
    )
    fig.colorbar(image, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    return fig, ax


def main():
    geometry, solution = solve_diel_cylinder()
    plot_diel_cylinder(geometry, solution)
    plt.show()


if __name__ == "__main__":
    main()
