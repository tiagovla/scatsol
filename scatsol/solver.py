from scatsol.constant import Polarization
from scatsol.geometry import CylindricalGeometry
from scatsol.geometry import SphericalGeometry
from scatsol.typing import FloatArray, ComplexArray, IntArray
from scipy.special import (
    h2vp,
    h1vp,
    hankel2,
    hankel1,
    jv,
    jvp,
    spherical_jn,
    spherical_yn,
    lpmv,
)
from typing import Annotated
import numpy as np


class CylindricalSolution:
    def __init__(
        self,
        n: int,
        an: ComplexArray,
        cn: ComplexArray,
        dn: ComplexArray,
        geometry: CylindricalGeometry,
        frequency: Annotated[float, "Hz"],
        polarization: Polarization = Polarization.TM,
    ):
        self.n = n
        self.an = an
        self.cn = cn
        self.dn = dn
        self.geometry = geometry
        self.frequency = frequency
        self.polarization = polarization

    @staticmethod
    def _points(xyz: FloatArray) -> FloatArray:
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz must have shape (number_of_points, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError("xyz must contain only finite coordinates.")
        return points

    def _fields_from_expansion(
        self,
        xyz: FloatArray,
        material,
        radial: ComplexArray,
        radial_derivative: ComplexArray,
        amplitudes: ComplexArray,
    ) -> tuple[ComplexArray, ComplexArray]:
        nn = np.arange(self.an.size)
        multiplicity = np.ones(nn.size)
        multiplicity[1:] = 2
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        cosine = np.cos(nn[:, None] * phi)
        sine = np.sin(nn[:, None] * phi)
        weighted = amplitudes * multiplicity

        scalar_field = weighted @ (radial * cosine)
        d_e_dphi = -(nn * weighted) @ (radial * sine)
        d_e_drho = weighted @ (material.k(self.frequency) * radial_derivative * cosine)
        nonzero = rho > 0
        e_field = np.zeros_like(xyz, dtype=np.complex128)
        h_field = np.zeros_like(xyz, dtype=np.complex128)
        if self.polarization == Polarization.TM:
            e_field[:, 2] = scalar_field
            omega_mu = 2 * np.pi * self.frequency * material.mu
            radial_field = np.zeros(rho.size, dtype=np.complex128)
            azimuthal_field = np.zeros(rho.size, dtype=np.complex128)
            radial_field[nonzero] = -d_e_dphi[nonzero] / (1j * omega_mu * rho[nonzero])
            azimuthal_field[nonzero] = d_e_drho[nonzero] / (1j * omega_mu)
            h_field[nonzero, 0] = radial_field[nonzero] * np.cos(phi[nonzero]) - azimuthal_field[
                nonzero
            ] * np.sin(phi[nonzero])
            h_field[nonzero, 1] = radial_field[nonzero] * np.sin(phi[nonzero]) + azimuthal_field[
                nonzero
            ] * np.cos(phi[nonzero])
            at_origin = ~nonzero
            if np.any(at_origin) and nn.size > 1:
                # Only the n=1 regular term has a nonzero gradient at the axis.
                h_field[at_origin, 1] = d_e_drho[at_origin] / (1j * omega_mu)
        else:
            h_field[:, 2] = scalar_field
            omega_epsilon = 2 * np.pi * self.frequency * material.epsilon
            radial_field = np.zeros(rho.size, dtype=np.complex128)
            azimuthal_field = np.zeros(rho.size, dtype=np.complex128)
            radial_field[nonzero] = d_e_dphi[nonzero] / (1j * omega_epsilon * rho[nonzero])
            azimuthal_field[nonzero] = -d_e_drho[nonzero] / (1j * omega_epsilon)
            e_field[nonzero, 0] = radial_field[nonzero] * np.cos(phi[nonzero]) - azimuthal_field[
                nonzero
            ] * np.sin(phi[nonzero])
            e_field[nonzero, 1] = radial_field[nonzero] * np.sin(phi[nonzero]) + azimuthal_field[
                nonzero
            ] * np.cos(phi[nonzero])
            at_origin = ~nonzero
            if np.any(at_origin) and nn.size > 1:
                e_field[at_origin, 1] = -d_e_drho[at_origin] / (1j * omega_epsilon)
        return e_field, h_field

    def total_field(self, xyz: FloatArray) -> tuple[ComplexArray, ComplexArray]:
        points = self._points(xyz)
        rho = np.hypot(points[:, 0], points[:, 1])
        nn = np.arange(self.an.size)
        e_field = np.zeros_like(points, dtype=np.complex128)
        h_field = np.zeros_like(points, dtype=np.complex128)

        for idx, region in enumerate(self.geometry.regions):
            mask = (rho >= region.inner_radius) & (rho < region.outer_radius)
            if not np.any(mask):
                continue
            if region.material.is_pec:
                continue
            k = region.material.k(self.frequency)
            argument = k * rho[mask]
            if idx == len(self.geometry.regions) - 1:
                # The mode count is chosen from the scatterer's size. It may
                # be too small to represent the incident plane wave far from
                # the cylinder, so expand only the scattered field here and
                # add the incident plane wave in closed form below.
                radial = np.zeros((nn.size, argument.size), dtype=np.complex128)
                radial_derivative = np.zeros_like(radial)
                active = self.an != 0
                if np.any(active):
                    radial[active] = self.an[active, None] * hankel2(nn[active, None], argument)
                    radial_derivative[active] = self.an[active, None] * h2vp(
                        nn[active, None], argument
                    )
                amplitudes = np.ones(nn.size, dtype=np.complex128)
            elif idx == 0:
                amplitudes = self.cn[:, 0] + self.dn[:, 0]
                radial = jv(nn[:, None], argument)
                radial_derivative = jvp(nn[:, None], argument)
            else:
                amplitudes = self.cn[:, idx]
                radial = amplitudes[:, None] * hankel1(nn[:, None], argument)
                radial += self.dn[:, idx, None] * hankel2(nn[:, None], argument)
                radial_derivative = amplitudes[:, None] * h1vp(nn[:, None], argument)
                radial_derivative += self.dn[:, idx, None] * h2vp(nn[:, None], argument)
                amplitudes = np.ones(nn.size, dtype=np.complex128)

            e_region, h_region = self._fields_from_expansion(
                points[mask], region.material, radial, radial_derivative, amplitudes
            )
            if idx == len(self.geometry.regions) - 1:
                incident_e, incident_h = self._incident_plane_wave(points[mask], region.material)
                e_region += incident_e
                h_region += incident_h
            e_field[mask] = e_region
            h_field[mask] = h_region
        return e_field, h_field

    def e_x(self, xyz: FloatArray) -> ComplexArray:
        """Return the x component of the total electric field at ``xyz``."""
        e_field, _ = self.total_field(xyz)
        return e_field[:, 0]

    def e_y(self, xyz: FloatArray) -> ComplexArray:
        """Return the y component of the total electric field at ``xyz``."""
        e_field, _ = self.total_field(xyz)
        return e_field[:, 1]

    def e_z(self, xyz: FloatArray) -> ComplexArray:
        """Return the z component of the total electric field at ``xyz``."""
        e_field, _ = self.total_field(xyz)
        return e_field[:, 2]

    def h_x(self, xyz: FloatArray) -> ComplexArray:
        """Return the x component of the total magnetic field at ``xyz``."""
        _, h_field = self.total_field(xyz)
        return h_field[:, 0]

    def h_y(self, xyz: FloatArray) -> ComplexArray:
        """Return the y component of the total magnetic field at ``xyz``."""
        _, h_field = self.total_field(xyz)
        return h_field[:, 1]

    def h_z(self, xyz: FloatArray) -> ComplexArray:
        """Return the z component of the total magnetic field at ``xyz``."""
        _, h_field = self.total_field(xyz)
        return h_field[:, 2]

    def incident_field(self, xyz: FloatArray) -> tuple[ComplexArray, ComplexArray]:
        points = self._points(xyz)
        return self._incident_plane_wave(points, self.geometry.regions[-1].material)

    def _incident_plane_wave(self, xyz: FloatArray, material):
        """Return the exact +x-directed incident plane wave in Cartesian form."""
        phase = np.exp(-1j * material.k(self.frequency) * xyz[:, 0])
        e_field = np.zeros_like(xyz, dtype=np.complex128)
        h_field = np.zeros_like(xyz, dtype=np.complex128)
        if self.polarization == Polarization.TM:
            e_field[:, 2] = phase
            h_field[:, 1] = -phase / material.eta
        else:
            h_field[:, 2] = phase
            e_field[:, 1] = material.eta * phase
        return e_field, h_field


class CylindricalSolver:
    def __init__(self, geometry: CylindricalGeometry, frequency: Annotated[float, "Hz"] = 300e6):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be a finite, positive value in hertz.")
        self.geometry = geometry
        self.frequency = frequency

    def __str__(self) -> str:
        return f"CylindricalSolver(geometry={self.geometry}, frequency={self.frequency})"

    def _calculate_coefficients(
        self, nn: IntArray, polarization: Polarization = Polarization.TM
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        n_regions = len(self.geometry.regions)
        if self.geometry.regions[0].material.is_pec:
            return self._calculate_coefficients_pec(nn, polarization)
        if n_regions == 1:
            return (
                np.zeros(nn.size, dtype=np.complex128),
                np.zeros((nn.size, 0), dtype=np.complex128),
                np.zeros((nn.size, 0), dtype=np.complex128),
            )
        radius = np.array([region.outer_radius for region in self.geometry.regions[:-1]])

        re = np.zeros((nn.size, n_regions - 1), dtype=np.complex128)
        dcn = np.zeros((nn.size, n_regions - 1), dtype=np.complex128)
        an = np.zeros(nn.size, dtype=np.complex128)
        cn = np.zeros((nn.size, n_regions - 1), dtype=np.complex128)
        dn = np.zeros((nn.size, n_regions - 1), dtype=np.complex128)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for i in range(n_regions - 1):
                mi, mip1 = self.geometry.regions[i].material, self.geometry.regions[i + 1].material
                ki, kip1 = mi.k(self.frequency), mip1.k(self.frequency)
                kiai, kip1ai = ki * radius[i], kip1 * radius[i]

                if i == 0:
                    dcn[:, i] = 1

                re[:, i] = self._calculate_re(nn, kiai, mi, mip1, dcn[:, i], polarization)
                if i < n_regions - 2:
                    dcn[:, i + 1] = self._calculate_dcn(nn, kip1ai, re[:, i])

            an = self._calculate_an(nn, radius, re)
            cn, dn = self._calculate_cn_dn(nn, radius, dcn, an)

        if not all(np.all(np.isfinite(values)) for values in (an, cn, dn)):
            raise FloatingPointError(
                "Cylindrical coefficients are non-finite; reduce n or use a better-conditioned "
                "geometry/material model."
            )

        return an, cn, dn

    def _calculate_coefficients_pec(
        self, nn: IntArray, polarization: Polarization
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        """Calculate coefficients for a PEC core with finite-material layers."""
        regions = self.geometry.regions
        n_regions = len(regions)
        if n_regions < 2:
            raise ValueError("A PEC core must be surrounded by at least one finite region.")
        if any(region.material.is_pec for region in regions[1:]):
            raise NotImplementedError("PEC is supported only for the innermost cylindrical region.")

        radii = np.array([region.outer_radius for region in regions[:-1]])
        n_modes = nn.size
        an = np.zeros(n_modes, dtype=np.complex128)
        cn = np.zeros((n_modes, n_regions - 1), dtype=np.complex128)
        dn = np.zeros_like(cn)

        first_finite_region = regions[1]
        k_first = first_finite_region.material.k(self.frequency)
        inner_argument = k_first * radii[0]
        dcn = np.zeros((n_modes, n_regions - 1), dtype=np.complex128)
        re = np.zeros_like(dcn)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if n_regions > 2:
                if polarization == Polarization.TM:
                    dcn[:, 1] = -hankel1(nn, inner_argument) / hankel2(nn, inner_argument)
                else:
                    dcn[:, 1] = -h1vp(nn, inner_argument) / h2vp(nn, inner_argument)

            for i in range(1, n_regions - 1):
                mi, mip1 = regions[i].material, regions[i + 1].material
                ki, kip1 = mi.k(self.frequency), mip1.k(self.frequency)
                re[:, i] = self._calculate_re(nn, ki * radii[i], mi, mip1, dcn[:, i], polarization)
                if i < n_regions - 2:
                    dcn[:, i + 1] = self._calculate_dcn(nn, kip1 * radii[i + 1], re[:, i])

            if n_regions == 2:
                k_out = regions[-1].material.k(self.frequency)
                argument = k_out * radii[0]
                if polarization == Polarization.TM:
                    numerator = jv(nn, argument)
                    denominator = hankel2(nn, argument)
                else:
                    numerator = jvp(nn, argument)
                    denominator = h2vp(nn, argument)
                an = -(1j ** (-nn)) * numerator / denominator
            else:
                an = self._calculate_an(nn, radii, re)
                kmam = regions[-2].material.k(self.frequency) * radii[-1]
                kam = regions[-1].material.k(self.frequency) * radii[-1]
                numerator = 1j ** (-nn) * jv(nn, kam) + an * hankel2(nn, kam)
                denominator = hankel1(nn, kmam) + dcn[:, -1] * hankel2(nn, kmam)
                cn[:, -1] = numerator / denominator
                dn[:, -1] = dcn[:, -1] * cn[:, -1]

                for i in range(n_regions - 3, 0, -1):
                    ki, kip1 = regions[i].material.k(self.frequency), regions[i + 1].material.k(
                        self.frequency
                    )
                    left_argument, right_argument = ki * radii[i], kip1 * radii[i]
                    numerator = cn[:, i + 1] * hankel1(nn, right_argument) + dn[:, i + 1] * hankel2(
                        nn, right_argument
                    )
                    denominator = hankel1(nn, left_argument) + dcn[:, i] * hankel2(
                        nn, left_argument
                    )
                    cn[:, i] = numerator / denominator
                    dn[:, i] = dcn[:, i] * cn[:, i]

        if not all(np.all(np.isfinite(values)) for values in (an, cn, dn)):
            raise FloatingPointError("PEC-cylinder coefficients are non-finite.")
        return an, cn, dn

    def _calculate_re(self, nn, kiai, mi, mip1, dcn_i, polarization=Polarization.TM):
        if polarization == Polarization.TM:
            numerator, denominator = mip1.epsilon * mi.mu, mip1.mu * mi.epsilon
        else:
            numerator, denominator = mip1.mu * mi.epsilon, mip1.epsilon * mi.mu
        amp = np.lib.scimath.sqrt(numerator / denominator)
        num = hankel1(nn, kiai) + dcn_i * hankel2(nn, kiai)
        den = h1vp(nn, kiai) + dcn_i * h2vp(nn, kiai)

        return amp * num / den

    def _calculate_dcn(self, nn, kip1ai, re_i):
        num = hankel1(nn, kip1ai) - re_i * h1vp(nn, kip1ai)
        den = hankel2(nn, kip1ai) - re_i * h2vp(nn, kip1ai)

        return -num / den

    def _calculate_an(
        self,
        nn: IntArray,
        radius: FloatArray,
        re: ComplexArray,
    ):
        an = np.zeros(nn.size, dtype=np.complex128)
        kam = self.geometry.regions[-1].material.k(self.frequency) * radius[-1]
        num = jv(nn, kam) - re[:, -1] * jvp(nn, kam)
        den = hankel2(nn, kam) - re[:, -1] * h2vp(nn, kam)
        an = -(1j ** (-nn)) * num / den

        return an

    def _calculate_cn_dn(
        self,
        nn: IntArray,
        radius: FloatArray,
        dcn: ComplexArray,
        an: ComplexArray,
    ) -> tuple[ComplexArray, ComplexArray]:
        n_regions = len(self.geometry.regions)
        cn = np.zeros((an.size, n_regions - 1), dtype=np.complex128)
        dn = np.zeros((an.size, n_regions - 1), dtype=np.complex128)

        kam = self.geometry.regions[-1].material.k(self.frequency) * radius[-1]
        kmam = self.geometry.regions[-2].material.k(self.frequency) * radius[-1]

        num = 1j ** (-nn) * jv(nn, kam) + an * hankel2(nn, kam)
        den = hankel1(nn, kmam) + dcn[:, -1] * hankel2(nn, kmam)
        cn[:, -1] = num / den
        dn[:, -1] = dcn[:, -1] * cn[:, -1]

        for i in range(n_regions - 3, -1, -1):
            mi, mip1 = self.geometry.regions[i].material, self.geometry.regions[i + 1].material
            ki, kip1 = mi.k(self.frequency), mip1.k(self.frequency)
            kiai, kip1ai = ki * radius[i], kip1 * radius[i]

            num = cn[:, i + 1] * hankel1(nn, kip1ai) + dn[:, i + 1] * hankel2(nn, kip1ai)
            den = hankel1(nn, kiai) + dcn[:, i] * hankel2(nn, kiai)
            cn[:, i] = num / den
            dn[:, i] = dcn[:, i] * cn[:, i]

        return cn, dn

    def solve(
        self,
        n: int = 50,
        polarization: Polarization = Polarization.TM,
    ) -> CylindricalSolution:
        """Solve for up to ``n`` cylindrical modes."""
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer number of modes.")

        if polarization not in (Polarization.TM, Polarization.TE):
            raise ValueError(f"Unsupported polarization: {polarization!r}")

        # Orders well above the largest size parameter contribute negligibly,
        # but direct Hankel evaluations become ill-conditioned there.
        interface_sizes = []
        for index, region in enumerate(self.geometry.regions[:-1]):
            adjacent = (region.material, self.geometry.regions[index + 1].material)
            finite_media = [material for material in adjacent if not material.is_pec]
            if finite_media:
                interface_sizes.append(
                    max(
                        abs(material.k(self.frequency) * region.outer_radius)
                        for material in finite_media
                    )
                )
        if interface_sizes:
            size_parameter = max(interface_sizes)
            n_required = int(np.ceil(size_parameter + 4 * size_parameter ** (1 / 3) + 2))
        else:
            n_required = n
        n_terms = min(n, n_required)
        an, cn, dn = self._calculate_coefficients(np.arange(0, n_terms), polarization)
        return CylindricalSolution(
            n=n_terms,
            an=an,
            cn=cn,
            dn=dn,
            geometry=self.geometry,
            frequency=self.frequency,
            polarization=polarization,
        )


def _spherical_radial(order, argument, kind):
    if kind == "j":
        value = spherical_jn(order, argument)
        derivative = spherical_jn(order, argument, derivative=True)
    elif kind == "y":
        value = spherical_yn(order, argument)
        derivative = spherical_yn(order, argument, derivative=True)
    else:
        value = spherical_jn(order, argument) - 1j * spherical_yn(order, argument)
        derivative = spherical_jn(order, argument, derivative=True) - 1j * spherical_yn(
            order, argument, derivative=True
        )
    riccati = argument * value
    riccati_derivative = value + argument * derivative
    return riccati, riccati_derivative


class SphericalSolution:
    """Fields and modal coefficients for concentric spherical regions."""

    def __init__(self, geometry, frequency, n, electric, magnetic, interior):
        self.geometry = geometry
        self.frequency = frequency
        self.n = n
        self.an, self.bn = electric, magnetic
        # Each interior entry is (regular-wave coefficients, singular-wave coefficients).
        self.interior = interior

    @staticmethod
    def _points(xyz):
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz must have shape (number_of_points, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError("xyz must contain only finite coordinates.")
        return points

    @staticmethod
    def _cartesian(spherical, points):
        _, theta, phi = points.T
        st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        result = np.empty_like(spherical)
        result[:, 0] = spherical[:, 0] * st * cp + spherical[:, 1] * ct * cp - spherical[:, 2] * sp
        result[:, 1] = spherical[:, 0] * st * sp + spherical[:, 1] * ct * sp + spherical[:, 2] * cp
        result[:, 2] = spherical[:, 0] * ct - spherical[:, 1] * st
        return result

    def _evaluate(self, xyz, material, radial_e, derivative_e, radial_m, derivative_m):
        points = xyz.copy()
        r = np.linalg.norm(points, axis=1)
        origin = r == 0
        if np.any(origin):
            points[origin, 2] = 1e-12 / max(abs(material.k(self.frequency)), 1.0)
        r, theta, phi = _cartesian_to_spherical(points)
        orders = np.arange(1, self.n + 1)
        x = material.k(self.frequency) * r
        p = np.array([lpmv(1, ell, np.cos(theta)) for ell in orders])
        # Associated-Legendre angular functions and their theta derivatives.
        ps = np.zeros_like(p)
        ps[0] = -1.0
        if self.n > 1:
            ps[1] = -3.0 * np.cos(theta)
        for idx in range(2, self.n):
            ell = idx + 1
            ps[idx] = (2 * ell - 1) / (ell - 1) * np.cos(theta) * ps[idx - 1] - ell / (
                ell - 1
            ) * ps[idx - 2]
        dps = np.zeros_like(p)
        dps[0] = np.cos(theta)
        for ell in range(2, self.n):
            dps[ell - 1] = (ell + 1) * ps[ell - 2] - ell * np.cos(theta) * ps[ell - 1]
        dps = -dps

        xe = np.asarray(x, dtype=np.complex128)[None, :]
        invx = np.divide(1.0, xe, out=np.zeros_like(xe, dtype=np.complex128), where=xe != 0)
        invx2 = invx**2
        e = np.zeros((xyz.shape[0], 3), dtype=np.complex128)
        h = np.zeros_like(e)
        lfac = orders * (orders + 1)
        cp, sp = np.cos(phi), np.sin(phi)
        e[:, 0] = cp * invx2[0] / 1j * (lfac[:, None] * radial_e * p).sum(axis=0)
        e[:, 1] = (
            -cp * invx[0] * (1j * (derivative_e * dps).sum(axis=0) + (radial_m * ps).sum(axis=0))
        )
        e[:, 2] = (
            sp * invx[0] * (1j * (derivative_e * ps).sum(axis=0) + (radial_m * dps).sum(axis=0))
        )
        h[:, 0] = sp * invx2[0] / 1j * (lfac[:, None] * radial_m * p).sum(axis=0)
        h[:, 1] = (
            -sp * invx[0] * (1j * (derivative_m * dps).sum(axis=0) + (radial_e * ps).sum(axis=0))
        )
        h[:, 2] = (
            -cp * invx[0] * (1j * (derivative_m * ps).sum(axis=0) + (radial_e * dps).sum(axis=0))
        )
        h /= material.eta
        spherical_points = np.column_stack((r, theta, phi))
        e = self._cartesian(e, spherical_points)
        h = self._cartesian(h, spherical_points)
        return e, h

    def total_field(self, xyz):
        points = self._points(xyz)
        radii = np.linalg.norm(points, axis=1)
        electric = np.zeros_like(points, dtype=np.complex128)
        magnetic = np.zeros_like(electric)
        orders = np.arange(1, self.n + 1)
        for idx, region in enumerate(self.geometry.regions):
            mask = (radii >= region.inner_radius) & (radii < region.outer_radius)
            if not np.any(mask) or region.material.is_pec:
                continue
            material = region.material
            selected_points = points[mask].copy()
            rr = radii[mask].copy()
            at_origin = rr == 0
            if np.any(at_origin):
                selected_points[at_origin, 2] = 1e-12 / max(abs(material.k(self.frequency)), 1.0)
                rr[at_origin] = np.linalg.norm(selected_points[at_origin], axis=1)
            arg = material.k(self.frequency) * rr
            if idx == len(self.geometry.regions) - 1:
                re = np.zeros((self.n, rr.size), dtype=np.complex128)
                de = np.zeros_like(re)
                rm = np.zeros_like(re)
                dm = np.zeros_like(re)
                active_e, active_m = self.an != 0, self.bn != 0
                if np.any(active_e) or np.any(active_m):
                    h2, h2p = _spherical_radial(orders[:, None], arg, "h2")
                    if np.any(active_e):
                        re[active_e] += self.an[active_e, None] * h2[active_e]
                        de[active_e] += self.an[active_e, None] * h2p[active_e]
                    if np.any(active_m):
                        rm[active_m] += self.bn[active_m, None] * h2[active_m]
                        dm[active_m] += self.bn[active_m, None] * h2p[active_m]
                ef, hf = self._evaluate(selected_points, material, re, de, rm, dm)
                incident_e, incident_h = self.incident_field(selected_points)
                electric[mask], magnetic[mask] = ef + incident_e, hf + incident_h
                continue
            else:
                c, d = self.interior[idx]
                j, jp = _spherical_radial(orders[:, None], arg, "j")
                if idx == 0:
                    re, de = c[:, 0, None] * j, c[:, 0, None] * jp
                    rm, dm = d[:, 0, None] * j, d[:, 0, None] * jp
                else:
                    y, yp = _spherical_radial(orders[:, None], arg, "y")
                    re, de = (
                        c[:, 0, None] * j + c[:, 1, None] * y,
                        c[:, 0, None] * jp + c[:, 1, None] * yp,
                    )
                    rm, dm = (
                        d[:, 0, None] * j + d[:, 1, None] * y,
                        d[:, 0, None] * jp + d[:, 1, None] * yp,
                    )
            ef, hf = self._evaluate(selected_points, material, re, de, rm, dm)
            electric[mask], magnetic[mask] = ef, hf
        return electric, magnetic

    def incident_field(self, xyz):
        points = self._points(xyz)
        material = self.geometry.regions[-1].material
        r, theta, phi = _cartesian_to_spherical(points)
        phase = np.exp(-1j * material.k(self.frequency) * points[:, 2])
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        e_spherical = np.column_stack((st * cp * phase, ct * cp * phase, -sp * phase))
        h_spherical = np.column_stack(
            (
                st * sp * phase / material.eta,
                ct * sp * phase / material.eta,
                cp * phase / material.eta,
            )
        )
        spherical_points = np.column_stack((r, theta, phi))
        return self._cartesian(e_spherical, spherical_points), self._cartesian(
            h_spherical, spherical_points
        )

    def e_x(self, xyz):
        return self.total_field(xyz)[0][:, 0]

    def e_y(self, xyz):
        return self.total_field(xyz)[0][:, 1]

    def e_z(self, xyz):
        return self.total_field(xyz)[0][:, 2]

    def h_x(self, xyz):
        return self.total_field(xyz)[1][:, 0]

    def h_y(self, xyz):
        return self.total_field(xyz)[1][:, 1]

    def h_z(self, xyz):
        return self.total_field(xyz)[1][:, 2]


def _cartesian_to_spherical(points):
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(np.hypot(points[:, 0], points[:, 1]), points[:, 2])
    phi = np.arctan2(points[:, 1], points[:, 0])
    return r, theta, phi


class SphericalSolver:
    """Mie solver for any number of concentric spherical material regions."""

    def __init__(self, geometry: SphericalGeometry, frequency: Annotated[float, "Hz"] = 300e6):
        if not isinstance(geometry, SphericalGeometry):
            raise TypeError("geometry must be a SphericalGeometry instance.")
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be a finite, positive value in hertz.")
        self.geometry, self.frequency = geometry, float(frequency)

    def _solve_channel(self, order, electric_channel):
        regions = self.geometry.regions
        count = len(regions)
        outer_radii = [region.outer_radius for region in regions[:-1]]
        pec_core = regions[0].material.is_pec
        unknown_slices = [None] * (count - 1)
        offset = 0
        for idx in range(count - 1):
            if idx == 0 and pec_core:
                unknown_slices[idx] = slice(offset, offset)
                continue
            width = 1 if idx == 0 else 2
            unknown_slices[idx] = slice(offset, offset + width)
            offset += width
        ext_index = offset
        size = ext_index + 1
        rows = 2 * (count - 1) - (1 if pec_core else 0)
        matrix = np.zeros((rows, size), dtype=np.complex128)
        rhs = np.zeros(rows, dtype=np.complex128)
        incident = (2 * order + 1) / (order * (order + 1)) * 1j ** (-order)
        row = 0
        if pec_core:
            radius = outer_radii[0]
            shell = regions[1]
            arg = shell.material.k(self.frequency) * radius
            if count == 2:
                fn, derivative = _spherical_radial(order, arg, "h2")
                inc_fn, inc_derivative = _spherical_radial(order, arg, "j")
                matrix[row, ext_index] = derivative if electric_channel else fn
                rhs[row] = -incident * (inc_derivative if electric_channel else inc_fn)
            else:
                for basis_idx, kind in enumerate(("j", "y")):
                    fn, derivative = _spherical_radial(order, arg, kind)
                    value = derivative if electric_channel else fn
                    matrix[row, unknown_slices[1].start + basis_idx] = value
            row += 1

        for boundary in range(1 if pec_core else 0, count - 1):
            radius = outer_radii[boundary]
            left, right = regions[boundary], regions[boundary + 1]
            if electric_channel:
                state_l = (1 / left.material.mu, 1 / left.material.k(self.frequency))
                state_r = (1 / right.material.mu, 1 / right.material.k(self.frequency))
            else:
                state_l = (1 / left.material.k(self.frequency), 1 / left.material.mu)
                state_r = (1 / right.material.k(self.frequency), 1 / right.material.mu)
            for derivative_row, weight_l, weight_r in (
                (False, state_l[0], state_r[0]),
                (True, state_l[1], state_r[1]),
            ):
                left_arg = left.material.k(self.frequency) * radius
                right_arg = right.material.k(self.frequency) * radius
                kinds_left = ["j"] if boundary == 0 else ["j", "y"]
                kinds_right = ["j", "y"] if boundary + 1 < count - 1 else ["h2"]
                for kind in kinds_left:
                    val, der = _spherical_radial(order, left_arg, kind)
                    coeff = der if derivative_row else val
                    if derivative_row:
                        coeff *= weight_l
                    else:
                        coeff *= weight_l
                    sl = unknown_slices[boundary]
                    matrix[row, sl.start + kinds_left.index(kind)] += coeff
                for kind in kinds_right:
                    val, der = _spherical_radial(order, right_arg, kind)
                    coeff = der if derivative_row else val
                    if derivative_row:
                        coeff *= -weight_r
                    else:
                        coeff *= -weight_r
                    if boundary + 1 == count - 1:
                        matrix[row, ext_index] += coeff
                        known, known_der = _spherical_radial(order, right_arg, "j")
                        known_coeff = known_der if derivative_row else known
                        if derivative_row:
                            known_coeff *= weight_r
                        else:
                            known_coeff *= weight_r
                        rhs[row] += known_coeff * incident
                    else:
                        sl = unknown_slices[boundary + 1]
                        matrix[row, sl.start + kinds_right.index(kind)] += coeff
                row += 1
        solution = np.linalg.solve(matrix, rhs)
        core_or_shell = []
        for idx, sl in enumerate(unknown_slices):
            if sl.stop == sl.start:
                core_or_shell.append(np.zeros(2, dtype=np.complex128))
            elif sl.stop - sl.start == 1:
                core_or_shell.append(np.array([solution[sl.start], 0j]))
            else:
                core_or_shell.append(solution[sl])
        return solution[ext_index], core_or_shell

    def solve(self, n: int = 50):
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer number of modes.")
        if any(region.material.is_pec for region in self.geometry.regions[1:]):
            raise NotImplementedError("PEC is supported only as the innermost spherical region.")
        if len(self.geometry.regions) == 1:
            zeros = np.zeros(n, dtype=np.complex128)
            return SphericalSolution(self.geometry, self.frequency, n, zeros, zeros.copy(), [])
        sizes = [
            abs(region.material.k(self.frequency) * region.outer_radius)
            for region in self.geometry.regions[:-1]
            if not region.material.is_pec
        ]
        if sizes:
            xmax = max(sizes)
            n = min(n, int(np.ceil(xmax + 4 * xmax ** (1 / 3) + 2)))
        an = np.zeros(n, dtype=np.complex128)
        bn = np.zeros_like(an)
        interior = [
            [np.zeros((n, 2), dtype=np.complex128), np.zeros((n, 2), dtype=np.complex128)]
            for _ in self.geometry.regions[:-1]
        ]
        for idx, order in enumerate(range(1, n + 1)):
            an[idx], ce = self._solve_channel(order, True)
            bn[idx], cm = self._solve_channel(order, False)
            for region_idx in range(len(interior)):
                interior[region_idx][0][idx] = ce[region_idx]
                interior[region_idx][1][idx] = cm[region_idx]
        if not all(
            np.all(np.isfinite(v)) for v in (an, bn, *[x for pair in interior for x in pair])
        ):
            raise FloatingPointError(
                "Spherical Mie coefficients are non-finite; reduce n or review the geometry."
            )
        return SphericalSolution(self.geometry, self.frequency, n, an, bn, interior)
