"""3D Mie scattering by a dielectric sphere."""

import numpy as np
from matplotlib import pyplot as plt

from scatsol import Material, SphericalGeometry, SphericalRegion, SphericalSolver


def solve_dielectric_sphere(frequency=300e6):
    """Return a dielectric sphere with a half-wavelength vacuum coating."""
    sphere_radius = 1.0
    vacuum = Material(epsilon_r=1.0, mu_r=1.0)
    wavelength = 2 * np.pi / abs(vacuum.k(frequency))
    coating_outer_radius = sphere_radius + 0.5 * wavelength
    geometry = SphericalGeometry(
        [
            SphericalRegion(0.0, sphere_radius, Material(epsilon_r=2.56, mu_r=1.0)),
            SphericalRegion(sphere_radius, coating_outer_radius, vacuum),
            SphericalRegion(coating_outer_radius, float("inf"), vacuum),
        ]
    )
    solution = SphericalSolver(geometry, frequency=frequency).solve(n=50)
    return geometry, solution


def plot_xz_field(geometry, solution, extent=3.0, resolution=80):
    """Plot the real eta0 H_y field in the y=0 plane, matching the thesis demo."""
    axis = np.linspace(-extent, extent, resolution)
    x, z = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    h_y = solution.h_y(points).real.reshape(x.shape) * Material.vacuum().eta

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.tripcolor(x.ravel(), z.ravel(), h_y.ravel(), shading="gouraud", vmin=-2, vmax=2)
    for region in geometry.regions[:-1]:
        ax.add_patch(plt.Circle((0, 0), region.outer_radius, fill=False, color="white"))
    fig.colorbar(image, ax=ax)
    ax.set(
        xlabel="x [m]",
        ylabel="z [m]",
        title=r"$\eta_0 H_y$",
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        aspect="equal",
    )
    ax.margins(0)
    return fig, ax


def main():
    geometry, solution = solve_dielectric_sphere()
    plot_xz_field(geometry, solution)
    plt.show()


if __name__ == "__main__":
    main()
