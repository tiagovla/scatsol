import numpy as np
import pytest

from scatsol import (
    CylindricalGeometry,
    CylindricalRegion,
    CylindricalSolver,
    Material,
    Polarization,
    __version__,
)
from scatsol.constant import SPEED_OF_LIGHT, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY


def test_version():
    assert __version__ == "0.0.2"


def test_vacuum_constants_are_consistent():
    assert VACUUM_PERMITTIVITY == pytest.approx(
        1 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2)
    )


def test_homogeneous_vacuum_matches_incident_plane_wave():
    vacuum = Material.vacuum()
    geometry = CylindricalGeometry([CylindricalRegion(0.0, float("inf"), vacuum)])
    solution = CylindricalSolver(geometry, frequency=300e6).solve(n=30)
    points = np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])

    electric, magnetic = solution.total_field(points)
    k = vacuum.k(300e6)
    expected_e = np.exp(-1j * k * points[:, 0])

    np.testing.assert_allclose(electric[:, 2], expected_e, atol=1e-12)
    np.testing.assert_allclose(magnetic[:, 1], -expected_e / vacuum.eta, atol=1e-12)
    assert np.all(np.isfinite(magnetic))
    np.testing.assert_array_equal(solution.e_x(points), electric[:, 0])
    np.testing.assert_array_equal(solution.e_y(points), electric[:, 1])
    np.testing.assert_array_equal(solution.e_z(points), electric[:, 2])
    np.testing.assert_array_equal(solution.h_x(points), magnetic[:, 0])
    np.testing.assert_array_equal(solution.h_y(points), magnetic[:, 1])
    np.testing.assert_array_equal(solution.h_z(points), magnetic[:, 2])


def test_tangential_fields_are_continuous_at_cylinder_interface():
    dielectric = Material(epsilon_r=4.0)
    vacuum = Material.vacuum()
    geometry = CylindricalGeometry(
        [
            CylindricalRegion(0.0, 0.5, dielectric),
            CylindricalRegion(0.5, float("inf"), vacuum),
        ]
    )
    solution = CylindricalSolver(geometry, frequency=300e6).solve(
        n=50, polarization=Polarization.TM
    )
    points = np.array([[0.5 - 1e-7, 0.0, 0.0], [0.5 + 1e-7, 0.0, 0.0]])
    electric, magnetic = solution.total_field(points)

    np.testing.assert_allclose(electric[0, 2], electric[1, 2], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(magnetic[0, 1], magnetic[1, 1], rtol=1e-5, atol=1e-7)


def test_negative_real_permittivity_uses_complex_wave_number():
    material = Material(epsilon_r=-2.0)
    assert np.isfinite(material.k(300e6))
    assert np.isfinite(material.eta)


@pytest.mark.parametrize(
    ("inner_radius", "outer_radius"),
    [(-1.0, 1.0), (1.0, 1.0), (1.0, float("nan")), (1.0, float("-inf"))],
)
def test_region_rejects_invalid_radii(inner_radius, outer_radius):
    with pytest.raises(ValueError):
        CylindricalRegion(inner_radius, outer_radius, Material.vacuum())


def test_geometry_rejects_empty_regions():
    with pytest.raises(ValueError, match="At least one region"):
        CylindricalGeometry([])


def test_solver_rejects_invalid_frequency_and_order():
    geometry = CylindricalGeometry([CylindricalRegion(0.0, float("inf"), Material.vacuum())])
    with pytest.raises(ValueError, match="frequency"):
        CylindricalSolver(geometry, frequency=0)
    solver = CylindricalSolver(geometry)
    with pytest.raises(ValueError, match="positive integer"):
        solver.solve(n=0)


def test_field_methods_reject_invalid_point_shape():
    geometry = CylindricalGeometry([CylindricalRegion(0.0, float("inf"), Material.vacuum())])
    solution = CylindricalSolver(geometry).solve(n=5)
    with pytest.raises(ValueError, match="shape"):
        solution.total_field(np.zeros((2, 2)))
