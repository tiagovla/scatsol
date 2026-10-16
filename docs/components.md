# API reference

All distances are in meters and frequencies are in hertz. Geometries list
regions from the center outward. Each region includes its inner radius and
excludes its outer radius; adjacent regions must share a boundary exactly.
The final region represents the exterior and must extend to positive infinity.

## Materials

`Material` takes relative permittivity and permeability. Either may be complex
for lossy or dispersive material models:

```python
from scatsol import Material

vacuum = Material.vacuum()
dielectric = Material(epsilon_r=2.5, mu_r=1.0)
pec = Material.PEC()
```

`epsilon`, `mu`, and `eta` return absolute permittivity, permeability, and wave
impedance. `k(frequency)` returns the wavenumber. `Material.PEC()` creates the
perfect electric conductor used for an innermost region.

::: scatsol.material.Material

## Cylindrical regions and geometry

The first cylindrical region starts at radius zero. Later regions form
concentric annuli; the exterior region ends at infinity.

```python
from scatsol import CylindricalGeometry, CylindricalRegion, Material

geometry = CylindricalGeometry([
    CylindricalRegion(0.0, 0.2, Material(epsilon_r=3.0)),
    CylindricalRegion(0.2, 0.4, Material(epsilon_r=2.0)),
    CylindricalRegion(0.4, float("inf"), Material.vacuum()),
])
```

::: scatsol.region.CylindricalRegion

::: scatsol.geometry.CylindricalGeometry

## Cylindrical solver and solution

The cylindrical solver models a plane wave traveling along +x. Select TM for
longitudinal `E_z`, or TE for longitudinal `H_z`:

```python
from scatsol import CylindricalSolver, Polarization

solution = CylindricalSolver(geometry, frequency=300e6).solve(
    n=50, polarization=Polarization.TE
)
```

`n` is the maximum number of modes requested. The solver may use fewer based on
the interface size parameters; `solution.n` gives the count used. PEC is
supported in the innermost region, with finite-material layers around it.

`total_field(points)` returns `(electric, magnetic)` complex arrays of shape
`(N, 3)`. Input points must have shape `(N, 3)` in Cartesian x, y, z order.
`incident_field(points)` returns the incident plane-wave fields in the exterior
material. Component methods such as `e_z(points)` and `h_y(points)` return
one-dimensional complex arrays.

::: scatsol.solver.CylindricalSolver

::: scatsol.solver.CylindricalSolution

## Spherical regions and geometry

Spherical regions are concentric shells. The first starts at radius zero, and
the exterior region extends to infinity.

```python
from scatsol import Material, SphericalGeometry, SphericalRegion

geometry = SphericalGeometry([
    SphericalRegion(0.0, 1.0, Material(epsilon_r=2.56)),
    SphericalRegion(1.0, float("inf"), Material.vacuum()),
])
```

::: scatsol.region.SphericalRegion

::: scatsol.geometry.SphericalGeometry

## Spherical solver and solution

The spherical solver uses an x-polarized plane wave traveling along +z. It
solves both electric and magnetic multipole channels for each order:

```python
from scatsol import SphericalSolver

solution = SphericalSolver(geometry, frequency=300e6).solve(n=50)
```

`n` is the maximum multipole order. As for the cylindrical solver, the mode
count may be reduced using the interface size parameters. The scattered
coefficients are available as `solution.an` (electric channel) and
`solution.bn` (magnetic channel). There is no `Polarization` argument because
both channels contribute to the returned fields.

The spherical `total_field`, `incident_field`, and component methods use the
same Cartesian input and output layout as the cylindrical solution. PEC is
supported as the innermost spherical region only.

::: scatsol.solver.SphericalSolver

::: scatsol.solver.SphericalSolution
