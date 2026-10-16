from scatsol.constant import Polarization
from scatsol.geometry import CylindricalGeometry
from scatsol.typing import FloatArray, ComplexArray, IntArray
from scipy.special import h2vp, h1vp, hankel2, hankel1, jv, jvp
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
    ):
        self.n = n
        self.an = an
        self.cn = cn
        self.dn = dn
        self.geometry = geometry
        self.frequency = frequency

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

        e_field = np.zeros_like(xyz, dtype=np.complex128)
        h_field = np.zeros_like(xyz, dtype=np.complex128)
        e_field[:, 2] = weighted @ (radial * cosine)

        d_e_dphi = -(nn * weighted) @ (radial * sine)
        d_e_drho = weighted @ (material.k(self.frequency) * radial_derivative * cosine)
        omega_mu = 2 * np.pi * self.frequency * material.mu
        nonzero = rho > 0
        h_rho = np.zeros(rho.size, dtype=np.complex128)
        h_phi = np.zeros(rho.size, dtype=np.complex128)
        h_rho[nonzero] = -d_e_dphi[nonzero] / (1j * omega_mu * rho[nonzero])
        h_phi[nonzero] = d_e_drho[nonzero] / (1j * omega_mu)
        h_field[nonzero, 0] = h_rho[nonzero] * np.cos(phi[nonzero]) - h_phi[nonzero] * np.sin(
            phi[nonzero]
        )
        h_field[nonzero, 1] = h_rho[nonzero] * np.sin(phi[nonzero]) + h_phi[nonzero] * np.cos(
            phi[nonzero]
        )

        at_origin = ~nonzero
        if np.any(at_origin) and nn.size > 1:
            # Only the n=1 regular term has a nonzero gradient at the axis.
            h_field[at_origin, 1] = d_e_drho[at_origin] / (1j * omega_mu)
        return e_field, h_field

    def total_field(self, xyz: FloatArray) -> tuple[ComplexArray, ComplexArray]:
        points = self._points(xyz)
        rho = np.hypot(points[:, 0], points[:, 1])
        nn = np.arange(self.an.size)
        incident_amplitudes = 1j**-nn
        e_field = np.zeros_like(points, dtype=np.complex128)
        h_field = np.zeros_like(points, dtype=np.complex128)

        for idx, region in enumerate(self.geometry.regions):
            mask = (rho >= region.inner_radius) & (rho < region.outer_radius)
            if not np.any(mask):
                continue
            k = region.material.k(self.frequency)
            argument = k * rho[mask]
            if idx == len(self.geometry.regions) - 1:
                radial = incident_amplitudes[:, None] * jv(nn[:, None], argument)
                radial_derivative = incident_amplitudes[:, None] * jvp(nn[:, None], argument)
                active = self.an != 0
                if np.any(active):
                    radial[active] += self.an[active, None] * hankel2(nn[active, None], argument)
                    radial_derivative[active] += self.an[active, None] * h2vp(
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
        k = self.geometry.regions[-1].material.k(self.frequency)
        nn = np.arange(self.an.size)
        amplitudes = 1j**-nn
        rho = np.hypot(points[:, 0], points[:, 1])
        argument = k * rho
        return self._fields_from_expansion(
            points,
            self.geometry.regions[-1].material,
            jv(nn[:, None], argument),
            jvp(nn[:, None], argument),
            amplitudes,
        )


class CylindricalSolver:
    def __init__(self, geometry: CylindricalGeometry, frequency: Annotated[float, "Hz"] = 300e6):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be a finite, positive value in hertz.")
        self.geometry = geometry
        self.frequency = frequency

    def __str__(self) -> str:
        return f"CylindricalSolver(geometry={self.geometry}, frequency={self.frequency})"

    def _calculate_coefficients_tm(
        self, nn: IntArray
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        n_regions = len(self.geometry.regions)
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

                re[:, i] = self._calculate_re(nn, kiai, mi, mip1, dcn[:, i])
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

    def _calculate_re(self, nn, kiai, mi, mip1, dcn_i):
        amp = np.lib.scimath.sqrt((mip1.epsilon * mi.mu) / (mip1.mu * mi.epsilon))
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

        if polarization == Polarization.TM:
            # Orders well above the largest size parameter contribute negligibly,
            # but direct Hankel evaluations become ill-conditioned there.
            interface_sizes = [
                abs(region.material.k(self.frequency) * region.outer_radius)
                for region in self.geometry.regions[:-1]
            ]
            if interface_sizes:
                size_parameter = max(interface_sizes)
                n_required = int(np.ceil(size_parameter + 4 * size_parameter ** (1 / 3) + 2))
            else:
                n_required = n
            n_terms = min(n, n_required)
            an, cn, dn = self._calculate_coefficients_tm(np.arange(0, n_terms))

            return CylindricalSolution(
                n=n_terms, an=an, cn=cn, dn=dn, geometry=self.geometry, frequency=self.frequency
            )
        else:
            raise NotImplementedError("Only TE polarization is implemented.")
