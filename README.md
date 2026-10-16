# ScatSol

ScatSol computes time-harmonic electromagnetic scattering from concentric
cylindrical and spherical regions. It returns complex electric and magnetic
fields at points you choose.

The cylindrical solver supports TM and TE plane-wave incidence, layered
materials, and a perfect electric conductor (PEC) core. The spherical solver
uses both electric and magnetic multipole channels for an x-polarized plane
wave. Materials can have real or complex relative permittivity and permeability.

## Install

Clone the repository, then install its dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quick start: cylinder

This example solves TM scattering from a dielectric cylinder in vacuum. Lengths
are in meters and frequency is in hertz.

```python
import numpy as np

from scatsol import (
    CylindricalGeometry,
    CylindricalRegion,
    CylindricalSolver,
    Material,
    Polarization,
)

vacuum = Material.vacuum()
dielectric = Material(epsilon_r=4.0)
geometry = CylindricalGeometry([
    CylindricalRegion(0.0, 0.5, dielectric),
    CylindricalRegion(0.5, float("inf"), vacuum),
])

solution = CylindricalSolver(geometry, frequency=300e6).solve(
    n=50, polarization=Polarization.TM
)
points = np.array([[0.75, 0.0, 0.0], [1.0, 0.5, 0.0]])
e_z = solution.e_z(points)
electric, magnetic = solution.total_field(points)
```

Points are Cartesian `(x, y, z)` coordinates in an `(N, 3)` array. Field
components and full fields are complex; use `.real`, `.imag`, or `np.abs(...)`
for the quantity you want to inspect.

## Quick start: sphere

The spherical solver returns both electric and magnetic multipole channels for
each solution:

```python
import numpy as np

from scatsol import Material, SphericalGeometry, SphericalRegion, SphericalSolver

vacuum = Material.vacuum()
geometry = SphericalGeometry([
    SphericalRegion(0.0, 1.0, Material(epsilon_r=2.56)),
    SphericalRegion(1.0, float("inf"), vacuum),
])
solution = SphericalSolver(geometry, frequency=300e6).solve(n=50)
points = np.array([[0.0, 0.0, 2.0], [1.5, 0.0, 0.0]])
magnetic_y = solution.h_y(points)
```

## Examples and documentation

Run a demonstration from the repository root, for example:

```bash
uv run python demo/cylinder_diel.py
uv run python demo/sphere_diel.py
```

The [documentation home](docs/index.md) includes the [equations and
conventions](docs/theory.md), [API reference](docs/components.md), and
[runnable examples](docs/examples.md). Build the documentation site with:

```bash
uv run mkdocs build
```
