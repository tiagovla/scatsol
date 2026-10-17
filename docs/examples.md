# Examples

`cylinder_cond.py`, `cylinder_diel.py`, `sphere_cond.py`, and `sphere_diel.py`
are based on cylinder and sphere examples in Jian-Ming Jin, *Theory and
Computation of Electromagnetic Fields*, 2nd ed. (Hoboken, NJ: Wiley-IEEE Press,
2015), adapted to the current geometry and solver APIs. The computed results
were compared with Jin's reported results and showed good agreement. The other
scripts are additional cylinder demonstrations.

## Conducting cylinder

`cylinder_cond.py` plots the TM `E_z` and TE `H_z` fields around a 1 m PEC
cylinder in vacuum at 300 MHz.

<iframe class="mpld3-embed" style="aspect-ratio: 5 / 2" src="../assets/cylinder_cond.html" title="Interactive TM electric and TE magnetic fields around a conducting cylinder" loading="lazy"></iframe>

## Basic dielectric cylinder

`cylinder_diel.py` plots the same two longitudinal field components for a 1 m
cylinder with relative permittivity 4 in vacuum at 300 MHz.

<iframe class="mpld3-embed" style="aspect-ratio: 5 / 2" src="../assets/cylinder_diel.html" title="Interactive TM electric and TE magnetic fields around a dielectric cylinder" loading="lazy"></iframe>

## Conducting sphere

`sphere_cond.py` plots the real `eta0 * H_y` field in the y=0 plane for a 1 m
PEC sphere in vacuum at 300 MHz. Both spherical multipole channels are included
by the solver.

<iframe class="mpld3-embed" style="aspect-ratio: 5 / 4" src="../assets/sphere_cond.html" title="Interactive magnetic field in the equatorial plane of a conducting sphere" loading="lazy"></iframe>

## Dielectric cylinder

`diel_cylinder.py` builds concentric cylindrical regions, solves the TM problem,
and plots the real part of `E_z`. Its `solve_diel_cylinder()` and
`plot_diel_cylinder(...)` functions can also be imported and used separately.

<iframe class="mpld3-embed" style="aspect-ratio: 5 / 4" src="../assets/diel_cylinder.html" title="Interactive electric field for the layered dielectric cylinder" loading="lazy"></iframe>

## PEC core with a dielectric coating

`pec_cylinder_coated.py` solves the same coated geometry for both TM and TE
polarization. It plots `E_z` for TM and `H_z` for TE. To run one polarization
from Python:

```python
from scatsol import Polarization
from demo.pec_cylinder_coated import solve_pec_cylinder_coated

geometry, solution = solve_pec_cylinder_coated(Polarization.TE)
```

The PEC core enforces zero tangential electric field at its surface. The demo
uses relative permittivity 2.5 for the coating, a 0.5 m core radius, a 0.8 m
outer coating radius, and a 300 MHz incident wave; edit those values in the
script to explore other cases.

<iframe class="mpld3-embed" style="aspect-ratio: 5 / 4" src="../assets/pec_cylinder_coated-1.html" title="Interactive TM electric field for a coated PEC cylinder" loading="lazy"></iframe>

<iframe class="mpld3-embed" style="aspect-ratio: 5 / 4" src="../assets/pec_cylinder_coated-2.html" title="Interactive TE magnetic field for a coated PEC cylinder" loading="lazy"></iframe>

## Dielectric sphere

`sphere_diel.py` implements the book’s dielectric sphere example using
`SphericalRegion`, `SphericalGeometry`, and `SphericalSolver`. It adds a vacuum
coating of thickness `0.5 λ` around a 1 m-radius sphere with relative
permittivity 2.56; at 300 MHz, the coating matches the exterior vacuum. It plots
the real `eta0 * H_y` field in the `y=0` plane. Its `solve_dielectric_sphere()`
and `plot_xz_field(...)` functions can also be imported independently.

<iframe class="mpld3-embed" style="aspect-ratio: 5 / 4" src="../assets/sphere_diel.html" title="Interactive magnetic field in the equatorial plane of a dielectric sphere" loading="lazy"></iframe>

## Tests

Run the numerical and input-validation checks with:

```bash
uv run pytest -q
```

To make warnings fail the run as well:

```bash
uv run pytest -q -W error
```
