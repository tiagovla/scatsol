# Examples

Runnable scripts live in the repository's `demo/` directory. Run them from the
repository root; each opens a Matplotlib figure.

```bash
uv run python demo/cylinder_cond.py
uv run python demo/cylinder_diel.py
uv run python demo/sphere_cond.py
uv run python demo/sphere_diel.py
uv run python demo/diel_cylinder.py
uv run python demo/pec_cylinder_coated.py
```

`cylinder_cond.py`, `cylinder_diel.py`, `sphere_cond.py`, and `sphere_diel.py`
recreate the four original `master` examples using the current geometry and
solver APIs. The other scripts are additional cylinder demonstrations.

## Conducting cylinder

`cylinder_cond.py` plots the TM `E_z` and TE `H_z` fields around a 1 m PEC
cylinder in vacuum at 300 MHz.

## Basic dielectric cylinder

`cylinder_diel.py` plots the same two longitudinal field components for a 1 m
cylinder with relative permittivity 4 in vacuum at 300 MHz.

## Conducting sphere

`sphere_cond.py` plots the real `eta0 * H_y` field in the y=0 plane for a 1 m
PEC sphere in vacuum at 300 MHz. Both spherical multipole channels are included
by the solver.

## Dielectric cylinder

`diel_cylinder.py` builds concentric cylindrical regions, solves the TM problem,
and plots the real part of `E_z`. Its `solve_diel_cylinder()` and
`plot_diel_cylinder(...)` functions can also be imported and used separately.

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

## Dielectric sphere

`sphere_diel.py` repeats the reference dielectric sphere example using
`SphericalRegion`, `SphericalGeometry`, and `SphericalSolver`. It adds a vacuum
coating of thickness `0.5 λ` around a 1 m-radius sphere with relative
permittivity 2.56; at 300 MHz, the coating matches the exterior vacuum. It plots
the real `eta0 * H_y` field in the `y=0` plane. Its `solve_dielectric_sphere()`
and `plot_xz_field(...)` functions can also be imported independently.

## Tests

Run the numerical and input-validation checks with:

```bash
uv run pytest -q
```

To make warnings fail the run as well:

```bash
uv run pytest -q -W error
```
