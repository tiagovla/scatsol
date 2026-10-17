"""TM and TE scattering by a dielectric cylinder."""

import matplotlib.pyplot as plt
import numpy as np

from scatsol import (
    CylindricalGeometry,
    CylindricalRegion,
    CylindricalSolver,
    Material,
    Polarization,
)


def solve_dielectric_cylinder(polarization=Polarization.TM):
    """Return the dielectric-cylinder geometry and its field solution."""
    radius = 1.0
    vacuum = Material.vacuum()
    dielectric = Material(epsilon_r=4.0, mu_r=1.0)
    geometry = CylindricalGeometry(
        [
            CylindricalRegion(0.0, radius, dielectric),
            CylindricalRegion(radius, float("inf"), vacuum),
        ]
    )
    solution = CylindricalSolver(geometry, frequency=300e6).solve(
        n=50, polarization=polarization
    )
    return geometry, solution


def plot_dielectric_cylinder(geometry, solutions, extent=3.0, resolution=100):
    """Plot the TM E_z and TE H_z fields side by side."""
    axis = np.linspace(-extent, extent, resolution)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    fields = [solutions[0].e_z(points).real, solutions[1].h_z(points).real]
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))
    for ax, field, title in zip(axes, fields, (r"$E_z$ - TMz", r"$H_z$ - TEz")):
        image = ax.tripcolor(
            points[:, 0], points[:, 1], field, shading="gouraud", vmin=-1, vmax=1
        )
        fig.colorbar(image, ax=ax)
        ax.add_patch(plt.Circle((0, 0), geometry.regions[0].outer_radius,
                                fill=False, color="white", linewidth=1))
        ax.set(xlabel="x [m]", ylabel="y [m]", title=title, xlim=(-extent, extent),
               ylim=(-extent, extent), aspect="equal")
        ax.margins(0)
    return fig, axes


def main():
    geometry, tm = solve_dielectric_cylinder(Polarization.TM)
    _, te = solve_dielectric_cylinder(Polarization.TE)
    plot_dielectric_cylinder(geometry, (tm, te))
    plt.show()


if __name__ == "__main__":
    main()
