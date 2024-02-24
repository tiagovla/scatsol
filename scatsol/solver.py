import numpy.typing as npt
from typing import Annotated, Tuple
from scatsol.constant import Polarization
from scatsol.geometry import CylindricalGeometry
from scipy.special import h2vp, h1vp, hankel2, hankel1, jv, jvp
import numpy as np


class CylindricalSolution:
    def __init__(
        self,
        n: int,
        an: npt.NDArray[np.complex128],
        geometry: CylindricalGeometry,
        frequency: Annotated[float, "Hz"],
    ):
        self.n = n
        self.an = an
        self.geometry = geometry
        self.frequency = frequency

    def total_field(self, xyz: npt.NDArray[np.float64]):
        e_field = np.zeros_like(xyz, dtype=np.complex128)
        h_field = np.zeros_like(xyz, dtype=np.complex128)

        raise NotImplementedError

        return e_field, h_field

    def incident_field(
        self, xyz: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        e_field = np.zeros_like(xyz, dtype=np.complex128)
        h_field = np.zeros_like(xyz, dtype=np.complex128)

        k = self.geometry.regions[-1].material.k(self.frequency)
        nn = np.arange(0, self.an.shape[0])
        eps = 1.0 + np.heaviside(nn - 0.5, 1)

        rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])

        besselterm = jv(nn[:, None], k * rho)
        costerm = np.cos(nn[:, None] * phi)
        an = 1j**-nn

        e_field[:, 2] = an * eps @ (besselterm * costerm)
        return e_field, h_field


class CylindricalSolver:
    def __init__(self, geometry: CylindricalGeometry, frequency: float = 300e6):
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
        dcn = np.zeros((n, n_regions))
        re = np.zeros((n, n_regions))
        an = np.zeros(n)
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

            num = hankel1(nn, kiai) + dcn[:, i] * hankel2(nn, kiai)
            den = h1vp(nn, kiai) + dcn[:, i] * h2vp(nn, kiai)

            re[:, i] = np.sqrt((mip1.epsilon * mi.mu) / (mip1.mu * mi.epsilon)) * num / den

            num = hankel1(nn, kip1ai) - re[:, i] * h1vp(nn, kip1ai)
            den = hankel2(nn, kip1ai) - re[:, i] * h2vp(nn, kip1ai)

            dcn[:, i + 1] = -num / den

        m = self.geometry.regions[-1].material
        k = m.k(self.frequency)
        kam = k * a[-1]
        num = jv(nn, kam) - re[:, -1] * jvp(nn, kam)
        den = hankel2(nn, kam) - re[:, -1] * h2vp(nn, kam)
        an = -(1j ** (-nn)) * num / den
        return CylindricalSolution(n=n, an=an, geometry=self.geometry, frequency=self.frequency)
