from typing import Callable
import numpy as np
import numpy.typing as npt
from scipy.special import h2vp, hankel2, jv, jvp

from scatsol.material import Material, Medium
import scatsol.utils


def mie_incident_field(
    xyz: npt.NDArray[np.float64],
    radius: float,
    frequency: float,
    background: Material,
    pol: str = "TM",
    *,
    n: int = 50,
):
    bg = Medium(background, frequency)
    kn = _an_inc(bg.k, radius, n)
    if pol == "TM":
        Ei, Hi = _calculate_field(xyz, bg.k, bg.eta, kn, jv, jvp)
    else:
        Hi, Ei = _calculate_field(xyz, bg.k, 1 / bg.eta, kn, jv, jvp)
        Ei = -Ei
    return Ei, Hi


def mie_total_field(
    xyz: npt.NDArray[np.float64],
    radius: float,
    frequency: float,
    background: Material,
    cylinder: Material | None = None,
    pol: str = "TM",
    *,
    n: int = 50,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:

    Erpz = np.zeros_like(xyz, dtype=np.complex128)
    Hrpz = np.zeros_like(xyz, dtype=np.complex128)
    mask = (xyz[:, :2] ** 2).sum(axis=1) > radius**2

    bg = Medium(background, frequency)
    if cylinder == None:  # if PEC cylinder
        if pol == "TM":
            an = _an_cond_tm(bg.k, radius, n)
            Erpz[mask], Hrpz[mask] = _calculate_field(xyz[mask], bg.k, bg.eta, an, hankel2, h2vp)
        else:
            an = _an_cond_te(bg.k, radius, n)
            Hrpz[mask], Erpz[mask] = _calculate_field(xyz[mask], bg.k, 1 / bg.eta, an, hankel2, h2vp)
            Erpz[mask] = -Erpz[mask]
    else:  # else diel cylinder
        c = Medium(cylinder, frequency)
        if pol == "TM":
            an, cn = _an_cn_diel_tm(bg.k, c.k, bg.eta, c.eta, radius, n)
            Erpz[mask], Hrpz[mask] = _calculate_field(xyz[mask], bg.k, bg.eta, an, hankel2, h2vp)
            Erpz[~mask], Hrpz[~mask] = _calculate_field(xyz[~mask], c.k, c.eta, cn, jv, jvp)
        else:
            an, cn = _an_cn_diel_tm(bg.k, c.k, 1 / bg.eta, 1 / c.eta, radius, n)
            Hrpz[mask], Erpz[mask] = _calculate_field(xyz[mask], bg.k, 1 / bg.eta, an, hankel2, h2vp)
            Hrpz[~mask], Erpz[~mask] = _calculate_field(xyz[~mask], c.k, 1 / c.eta, cn, jv, jvp)
            Erpz[mask] = -Erpz[mask]

    Ei, Hi = mie_incident_field(xyz[mask], radius, frequency, background, pol, n=n)
    Erpz[mask] += Ei
    Hrpz[mask] += Hi

    return Erpz, Hrpz


def _an_cond_tm(k: float | complex, a: float, n: int) -> npt.NDArray[np.complex128]:
    nn = np.arange(0, n)
    return -(1j**-nn) * jv(nn, k * a) / hankel2(nn, k * a)


def _an_cn_diel_tm(
    k: float | complex,
    kd: float | complex,
    eta: float | complex,
    etad: float | complex,
    a: float,
    n: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    nn = np.arange(0, n)
    den = eta * hankel2(nn, k * a) * jvp(nn, kd * a) - etad * h2vp(nn, k * a) * jv(nn, kd * a)
    num = eta * jv(nn, k * a) * jvp(nn, kd * a) - etad * jvp(nn, k * a) * jv(nn, kd * a)
    an = -(1j ** (-nn)) * num / den
    cn = (etad * 2 * (1j ** (-nn + 1))) / (np.pi * k * a * den)

    return an, cn


def _an_cond_te(k: float | complex, a: float, n: int) -> npt.NDArray[np.complex128]:
    nn = np.arange(0, n)
    return -(1j**-nn) * jvp(nn, k * a) / h2vp(nn, k * a)


def _an_inc(k: float | complex, a: float, n: int) -> npt.NDArray[np.complex128]:
    nn = np.arange(0, n)
    return 1j**-nn


def _calculate_field(
    xyz: npt.NDArray[np.float64],
    k: float | complex,
    eta: float | complex,
    an: npt.NDArray[np.complex128],
    fn: Callable,
    dfn: Callable,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    nn = np.arange(0, an.shape[0])
    eps = 1.0 + np.heaviside(nn - 0.5, 1)
    rho, phi, _ = scatsol.utils.cart2cyl(xyz).T

    e_field = np.zeros_like(xyz, dtype=np.complex128)
    h_field = np.zeros_like(xyz, dtype=np.complex128)

    besselterm = fn(nn[:, None], k * rho)
    besselpterm = dfn(nn[:, None], k * rho)
    costerm = np.cos(nn[:, None] * phi)
    sinterm = np.sin(nn[:, None] * phi)
    e_field[:, 2] = an * eps @ (besselterm * costerm)
    h_field[:, 0] = (nn * an * eps @ (besselterm * sinterm)) / (1j * k * eta * rho)
    h_field[:, 1] = (k * an * eps @ (besselpterm * costerm)) / (1j * k * eta * rho)

    return e_field, h_field
