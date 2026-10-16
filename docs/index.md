# ScatSol

ScatSol computes time-harmonic electromagnetic scattering by concentric
cylindrical and spherical regions. Give it a geometry, materials, and a
frequency; it returns complex electric and magnetic fields at requested points.

| Geometry | Incident field | Solver output |
| --- | --- | --- |
| Cylinder | Plane wave along +x; choose TM or TE | Fields with the requested polarization |
| Sphere | x-polarized plane wave along +z | Both electric and magnetic multipole channels |

Both solvers support dielectric and magnetic materials with real or complex
relative permittivity and permeability. PEC is supported as the innermost
region, with finite-material layers around it.

## Install

From the repository root, install the project dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quick start

This example solves TM scattering from a dielectric cylinder in vacuum. Lengths
are in meters, frequency is in hertz, and points use Cartesian `(x, y, z)`
coordinates.

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
geometry = CylindricalGeometry([
    CylindricalRegion(0.0, 0.5, Material(epsilon_r=4.0)),
    CylindricalRegion(0.5, float("inf"), vacuum),
])
solution = CylindricalSolver(geometry, frequency=300e6).solve(
    n=50, polarization=Polarization.TM
)

points = np.array([[0.75, 0.0, 0.0], [1.0, 0.5, 0.0]])
e_z = solution.e_z(points)
electric, magnetic = solution.total_field(points)
```

`total_field()` returns electric and magnetic arrays of shape `(N, 3)`, with
columns ordered x, y, z. Component methods such as `e_z(points)` return a
one-dimensional complex array. See [the API reference](components.md) for the
spherical solver and the other public classes.

## Learn more

- [Equations and conventions](theory.md): cylindrical TM/TE fields, spherical
  multipoles, interface conditions, and PEC boundaries.
- [API reference](components.md): materials, regions, geometries, solvers, and
  field methods.
- [Examples](examples.md): runnable scripts for dielectric and conducting
  cylinders and spheres.
