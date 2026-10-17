import numpy as np
import pytest

from scatsol import (
    CylindricalGeometry,
    CylindricalRegion,
    CylindricalSolver,
    Material,
    Polarization,
    SphericalGeometry,
    SphericalRegion,
    SphericalSolver,
)
from scatsol.constant import SPEED_OF_LIGHT, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY


def test_vacuum_constants_are_consistent():
    assert VACUUM_PERMITTIVITY == pytest.approx(1 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2))


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


def test_te_homogeneous_vacuum_matches_incident_plane_wave():
    vacuum = Material.vacuum()
    geometry = CylindricalGeometry([CylindricalRegion(0.0, float("inf"), vacuum)])
    solution = CylindricalSolver(geometry, frequency=300e6).solve(
        n=30, polarization=Polarization.TE
    )
    points = np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])

    electric, magnetic = solution.total_field(points)
    expected_h = np.exp(-1j * vacuum.k(300e6) * points[:, 0])

    np.testing.assert_allclose(magnetic[:, 2], expected_h, atol=1e-12)
    np.testing.assert_allclose(electric[:, 1], vacuum.eta * expected_h, atol=1e-12)


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


def test_te_tangential_fields_are_continuous_at_cylinder_interface():
    dielectric = Material(epsilon_r=4.0)
    vacuum = Material.vacuum()
    geometry = CylindricalGeometry(
        [
            CylindricalRegion(0.0, 0.5, dielectric),
            CylindricalRegion(0.5, float("inf"), vacuum),
        ]
    )
    solution = CylindricalSolver(geometry, frequency=300e6).solve(
        n=50, polarization=Polarization.TE
    )
    points = np.array([[0.5 - 1e-7, 0.0, 0.0], [0.5 + 1e-7, 0.0, 0.0]])
    electric, magnetic = solution.total_field(points)

    np.testing.assert_allclose(magnetic[0, 2], magnetic[1, 2], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(electric[0, 1], electric[1, 1], rtol=1e-5, atol=1e-7)


def test_pec_core_enforces_zero_tangential_electric_field():
    geometry = CylindricalGeometry(
        [
            CylindricalRegion(0.0, 0.5, Material.PEC()),
            CylindricalRegion(0.5, 0.8, Material(epsilon_r=2.5)),
            CylindricalRegion(0.8, float("inf"), Material.vacuum()),
        ]
    )
    solution = CylindricalSolver(geometry, frequency=300e6).solve(n=50)
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

    e_z = solution.e_z(points)
    assert e_z[0] == 0
    assert abs(e_z[1]) < 1e-12
    assert np.all(np.isfinite(e_z))


def test_te_pec_core_enforces_zero_tangential_electric_field():
    geometry = CylindricalGeometry(
        [
            CylindricalRegion(0.0, 0.5, Material.PEC()),
            CylindricalRegion(0.5, 0.8, Material(epsilon_r=2.5)),
            CylindricalRegion(0.8, float("inf"), Material.vacuum()),
        ]
    )
    solution = CylindricalSolver(geometry, frequency=300e6).solve(
        n=50, polarization=Polarization.TE
    )
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

    e_y = solution.e_y(points)
    assert e_y[0] == 0
    assert abs(e_y[1]) < 1e-10
    assert np.all(np.isfinite(e_y))


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


def test_spherical_homogeneous_medium_matches_incident_plane_wave():
    vacuum = Material.vacuum()
    geometry = SphericalGeometry([SphericalRegion(0.0, float("inf"), vacuum)])
    solution = SphericalSolver(geometry, frequency=300e6).solve(n=30)
    points = np.array([[0.2, 0.1, 0.3], [0.0, 0.0, 0.0], [-0.4, 0.2, -0.1]])

    electric, magnetic = solution.total_field(points)
    phase = np.exp(-1j * vacuum.k(300e6) * points[:, 2])
    np.testing.assert_allclose(electric[:, 0], phase, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(electric[:, 1:], 0.0, atol=1e-10)
    np.testing.assert_allclose(magnetic[:, 1], phase / vacuum.eta, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(magnetic[:, [0, 2]], 0.0, atol=1e-10)


def test_spherical_tangential_fields_are_continuous_at_material_interfaces():
    geometry = SphericalGeometry(
        [
            SphericalRegion(0.0, 0.4, Material(epsilon_r=3.0)),
            SphericalRegion(0.4, 0.7, Material(epsilon_r=2.0)),
            SphericalRegion(0.7, float("inf"), Material.vacuum()),
        ]
    )
    solution = SphericalSolver(geometry, frequency=300e6).solve(n=40)
    points = np.array([[0.0, 0.4 - 1e-7, 0.0], [0.0, 0.4 + 1e-7, 0.0]])
    electric, magnetic = solution.total_field(points)

    np.testing.assert_allclose(electric[0, [0, 2]], electric[1, [0, 2]], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(magnetic[0, [0, 2]], magnetic[1, [0, 2]], rtol=1e-5, atol=1e-7)


def test_spherical_pec_core_with_coating_has_finite_fields():
    geometry = SphericalGeometry(
        [
            SphericalRegion(0.0, 0.2, Material.PEC()),
            SphericalRegion(0.2, 0.35, Material(epsilon_r=2.5)),
            SphericalRegion(0.35, float("inf"), Material.vacuum()),
        ]
    )
    solution = SphericalSolver(geometry, frequency=300e6).solve(n=30)
    points = np.array([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0], [0.0, 0.3, 0.0], [0.0, 0.5, 0.0]])
    electric, magnetic = solution.total_field(points)

    assert np.all(electric[0] == 0)
    assert np.all(magnetic[0] == 0)
    np.testing.assert_allclose(electric[1, [0, 2]], 0.0, atol=1e-8)
    assert np.all(np.isfinite(electric))
    assert np.all(np.isfinite(magnetic))


def test_spherical_geometry_rejects_disconnected_regions():
    with pytest.raises(ValueError, match="contiguous"):
        SphericalGeometry(
            [
                SphericalRegion(0.0, 0.3, Material(epsilon_r=2.0)),
                SphericalRegion(0.4, float("inf"), Material.vacuum()),
            ]
        )


def test_spherical_coating_matching_exterior_preserves_uncoated_solution():
    vacuum = Material.vacuum()
    dielectric = Material(epsilon_r=2.56, mu_r=1.0)
    frequency = 300e6
    coating_outer_radius = 1.0 + 0.5 * (2 * np.pi / abs(vacuum.k(frequency)))
    bare = SphericalGeometry(
        [
            SphericalRegion(0.0, 1.0, dielectric),
            SphericalRegion(1.0, float("inf"), vacuum),
        ]
    )
    coated = SphericalGeometry(
        [
            SphericalRegion(0.0, 1.0, dielectric),
            SphericalRegion(1.0, coating_outer_radius, vacuum),
            SphericalRegion(coating_outer_radius, float("inf"), vacuum),
        ]
    )
    bare_solution = SphericalSolver(bare, frequency=frequency).solve(n=50)
    coated_solution = SphericalSolver(coated, frequency=frequency).solve(n=50)
    points = np.array([[1.5, 0.0, 0.2], [0.0, 1.8, 0.5], [0.0, 0.0, 2.0]])

    bare_fields = bare_solution.total_field(points)
    coated_fields = coated_solution.total_field(points)
    np.testing.assert_allclose(coated_fields[0], bare_fields[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(coated_fields[1], bare_fields[1], rtol=1e-9, atol=1e-9)


def test_spherical_dielectric_demo_matches_reference_field_value():
    vacuum = Material.vacuum()
    geometry = SphericalGeometry(
        [
            SphericalRegion(0.0, 1.0, Material(epsilon_r=2.56, mu_r=1.0)),
            SphericalRegion(1.0, float("inf"), vacuum),
        ]
    )
    solution = SphericalSolver(geometry, frequency=300e6).solve(n=50)
    point = np.array([[1.5, 0.0, 0.2]])

    eta_h_y = vacuum.eta * solution.h_y(point)[0]
    assert eta_h_y == pytest.approx(0.19067 - 0.79659j, abs=2e-5)
