import numpy.typing as npt
from typing import Annotated
from scatsol.constant import Polarization
from scatsol.geometry import CylindricalGeometry
from scipy.special import h2vp, h1vp, hankel2, hankel1, jv, jvp
import numpy as np


class CylindricalSolution:
    def __init__(
        self,
        n: int,
        an: npt.NDArray[np.complex128],
        cn: npt.NDArray[np.complex128],
        dn: npt.NDArray[np.complex128],
        geometry: CylindricalGeometry,
        frequency: Annotated[float, "Hz"],
    ):
        self.n = n
        self.an = an
        self.cn = cn
        self.dn = dn
        self.geometry = geometry
        self.frequency = frequency

    def total_field(
        self, xyz: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        e_field = np.zeros_like(xyz, dtype=np.complex128)
        h_field = np.zeros_like(xyz, dtype=np.complex128)

        nn = np.arange(0, self.an.shape[0])
        rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        eps = 1.0 + np.heaviside(nn - 0.5, 1)

        costerm = np.cos(nn[:, None] * phi)
        sinterm = np.sin(nn[:, None] * phi)

        for idx, region in enumerate(self.geometry.regions):
            k = region.material.k(self.frequency)
            mask = np.vectorize(region.check_within)(rho)
            if region == self.geometry.regions[-1]:
                besselterm = jv(nn[:, None], k * rho[mask])
                e_field[mask, 2] = ((self.an + 1j**-nn) * eps) @ (besselterm * costerm[:, mask])
            elif region == self.geometry.regions[0]:
                besselterm = jv(nn[:, None], k * rho[mask])
                e_field[mask, 2] = ((self.cn[:, 0] + self.dn[:, 0]) * eps) @ (
                    besselterm * costerm[:, mask]
                )
            else:
                besselterm_0 = hankel1(nn[:, None], k * rho[mask])
                besselterm_1 = hankel2(nn[:, None], k * rho[mask])
                e_field[mask, 2] = (self.cn[:, idx] * eps) @ (besselterm_0 * costerm[:, mask])
                e_field[mask, 2] += (self.dn[:, idx] * eps) @ (besselterm_1 * costerm[:, mask])
        return e_field, h_field

    def incident_field(
        self, xyz: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        e_field = np.zeros_like(xyz, dtype=np.complex128)
        h_field = np.zeros_like(xyz, dtype=np.complex128)

        k = self.geometry.regions[-1].material.k(self.frequency)
        eta = self.geometry.regions[-1].material.eta
        nn = np.arange(0, self.an.shape[0])
        eps = 1.0 + np.heaviside(nn - 0.5, 1)

        rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])

        besselterm = jv(nn[:, None], k * rho)
        besselpterm = jvp(nn[:, None], k * rho)
        costerm = np.cos(nn[:, None] * phi)
        sinterm = np.sin(nn[:, None] * phi)
        an = 1j**-nn

        e_field[:, 2] = an * eps @ (besselterm * costerm)
        h_field[:, 0] = (nn * an * eps @ (besselterm * sinterm)) / (1j * k * eta * rho)
        h_field[:, 1] = (k * an * eps @ (besselpterm * costerm)) / (1j * k * eta * rho)
        return e_field, h_field


class CylindricalSolver:
    def __init__(self, geometry: CylindricalGeometry, frequency: Annotated[float, "Hz"] = 300e6):
        self.geometry = geometry
        self.frequency = frequency

    def __str__(self) -> str:
        return f"CylindricalSolver(geometry={self.geometry}, frequency={self.frequency})"

    def solve(
        self,
        n: int = 50,
        tolerance: float | None = 1e-6,
        polarization: Polarization = Polarization.TM,
    ):
        if n is None and tolerance is None:
            raise ValueError("n or tolerance must be provided.")

        n_regions = len(self.geometry.regions)
        a = np.array([region.outer_radius for region in self.geometry.regions[:-1]])
        dcn = np.zeros((n, n_regions), dtype=np.complex128)
        cn = np.zeros((n, n_regions), dtype=np.complex128)
        dn = np.zeros((n, n_regions), dtype=np.complex128)
        re = np.zeros((n, n_regions), dtype=np.complex128)
        an = np.zeros(n, dtype=np.complex128)
        nn = np.arange(0, n)

        for i in range(0, n_regions - 1):
            mi = self.geometry.regions[i].material
            mip1 = self.geometry.regions[i + 1].material
            ki = mi.k(self.frequency)
            kip1 = mip1.k(self.frequency)

            if i == 0:
                dcn[:, i] = 1

            kiai = ki * a[i]
            kip1ai = kip1 * a[i]

            amp = np.sqrt((mip1.epsilon * mi.mu) / (mip1.mu * mi.epsilon))
            num = hankel1(nn, kiai) + dcn[:, i] * hankel2(nn, kiai)
            den = h1vp(nn, kiai) + dcn[:, i] * h2vp(nn, kiai)

            re[:, i] = amp * num / den

            if i < n_regions - 1:
                num = hankel1(nn, kip1ai) - re[:, i] * h1vp(nn, kip1ai)
                den = hankel2(nn, kip1ai) - re[:, i] * h2vp(nn, kip1ai)
                dcn[:, i + 1] = -num / den

        kam = self.geometry.regions[-1].material.k(self.frequency) * a[-1]
        kmam = self.geometry.regions[-2].material.k(self.frequency) * a[-1]
        num = jv(nn, kam) - re[:, -2] * jvp(nn, kam)
        den = hankel2(nn, kam) - re[:, -2] * h2vp(nn, kam)
        an = -(1j ** (-nn)) * num / den

        # founding cn and dn by going backwards
        num = 1j ** (-nn) * jv(nn, kam) + an * hankel2(nn, kam)
        den = hankel1(nn, kmam) + dcn[:, -2] * hankel2(nn, kmam)
        cn[:, -2] = num / den
        dn[:, -2] = dcn[:, -2] * cn[:, -2]

        for i in range(n_regions - 3, -1, -1):
            mi = self.geometry.regions[i].material
            mip1 = self.geometry.regions[i + 1].material
            ki = mi.k(self.frequency)
            kip1 = mip1.k(self.frequency)
            kiai = ki * a[i]
            kip1ai = kip1 * a[i]

            num = cn[:, i + 1] * hankel1(nn, kip1ai) + dn[:, i + 1] * hankel2(nn, kip1ai)
            den = hankel1(nn, kiai) + dcn[:, i] * hankel2(nn, kiai)
            cn[:, i] = num / den
            dn[:, i] = dcn[:, i] * cn[:, i]

        cn = np.nan_to_num(cn)  # HACK: nan values should be correctly handled
        dn = np.nan_to_num(dn)

        return CylindricalSolution(
            n=n, an=an, cn=cn, dn=dn, geometry=self.geometry, frequency=self.frequency
        )
