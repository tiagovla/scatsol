"""3D Mie scattering by a perfectly conducting sphere."""

import matplotlib.pyplot as plt
import numpy as np

from scatsol import (
    Material,
    SphericalGeometry,
    SphericalRegion,
    SphericalSolver,
)


def solve_conducting_sphere(frequency=300e6):
    """Return the PEC-sphere geometry and its field solution."""
    radius = 1.0
    geometry = SphericalGeometry(
        [
            SphericalRegion(0.0, radius, Material.PEC()),
            SphericalRegion(radius, float("inf"), Material.vacuum()),
        ]
    )
    solution = SphericalSolver(geometry, frequency=frequency).solve(n=50)
    return geometry, solution


def plot_xz_field(geometry, solution, extent=5.0, resolution=50):
    """Plot real eta0 H_y in the y=0 plane."""
    axis = np.linspace(-extent, extent, resolution)
    x, z = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    h_y = Material.vacuum().eta * solution.h_y(points).real
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 4))
    image = ax.tripcolor(
        x.ravel(), z.ravel(), h_y, shading="gouraud", vmin=-2, vmax=2
    )
    fig.colorbar(image, ax=ax)
    ax.add_patch(plt.Circle((0, 0), geometry.regions[0].outer_radius,
                            fill=False, color="white", linewidth=1))
    ax.set(xlabel="x [m]", ylabel="z [m]", title=r"$\eta_0 H_y$",
           xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.margins(0)
    return fig, ax


def main():
    geometry, solution = solve_conducting_sphere()
    plot_xz_field(geometry, solution)
    plt.show()


if __name__ == "__main__":
    main()
