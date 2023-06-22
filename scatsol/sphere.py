import numpy as np
from typing import Callable
import numpy.typing as npt
from scipy.special import spherical_jn as sjn, spherical_yn as syn
from scatsol.material import Material, Medium
import scatsol.utils
from scatsol.utils import ric_jnp, ric_jn, ric_h2n, ric_h2np


def mie_incident_field(
    xyz: npt.NDArray[np.float64],
    frequency: float,
    background: Material,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    Ertp = np.zeros_like(xyz, dtype=np.complex128)
    Hrtp = np.zeros_like(xyz, dtype=np.complex128)
    r, theta, phi = scatsol.utils.cart2sph(xyz).T
    bg = Medium(background, frequency)
    t_cos, t_sin = np.cos(theta), np.sin(theta)
    p_cos, p_sin = np.cos(phi), np.sin(phi)
    exp = np.exp(-1j * bg.k * r * np.cos(theta))
    Ertp[:, 0] = t_sin * p_cos * exp
    Ertp[:, 1] = t_cos * p_cos * exp
    Ertp[:, 2] = -p_sin * exp
    Hrtp[:, 0] = t_sin * p_sin * exp / bg.eta
    Hrtp[:, 1] = t_cos * p_sin * exp / bg.eta
    Hrtp[:, 2] = p_cos * exp / bg.eta
    return Ertp, Hrtp


def mie_total_field(
    xyz: npt.NDArray[np.float64],
    radius: float,
    frequency: float,
    background: Material,
    sphere: Material | None = None,
    *,
    n: int = 50,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    Ertp = np.zeros_like(xyz, dtype=np.complex128)
    Hrtp = np.zeros_like(xyz, dtype=np.complex128)

    mask = (xyz**2).sum(axis=1) > radius**2

    bg = Medium(background, frequency)
    if sphere == None:
        an, bn = _an_bn_cond(bg.k, radius, n)
        Ertp[mask], Hrtp[mask] = _calculate_field(xyz[mask], bg.k, bg.eta, an, bn, ric_h2n, ric_h2np)
    else:
        s = Medium(sphere, frequency)
        an, bn, cn, dn = _an_bn_cn_dn_diel(bg.k, sphere.epsilon_r, sphere.mu_r, radius, n)
        Ertp[mask], Hrtp[mask] = _calculate_field(xyz[mask], bg.k, bg.eta, an, bn, ric_h2n, ric_h2np)
        Ertp[~mask], Hrtp[~mask] = _calculate_field(xyz[~mask], s.k, s.eta, cn, dn, ric_jn, ric_jnp)
    Ei, Hi = mie_incident_field(xyz[mask], frequency, background)
    Ertp[mask] += Ei
    Hrtp[mask] += Hi
    return Ertp, Hrtp


def _an_bn_cond(k: float | complex, a: float, n: int) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    nn = np.arange(1, n + 1)
    scale_term = (2 * nn + 1) / (nn * (nn + 1))
    sj = sjn(nn, k * a)
    sy = syn(nn, k * a)
    sjp = sjn(nn, k * a, derivative=True)
    syp = syn(nn, k * a, derivative=True)
    bn = -(1j**-nn) * scale_term * sj / (sj - 1j * sy)
    an_num = sj + k * a * sjp
    an_den = an_num - 1j * (sy + k * a * syp)
    an = -(1j**-nn) * scale_term * an_num / an_den
    return an, bn


def _an_bn_cn_dn_diel(
    k_0: float | complex,
    eps_r: float | complex,
    mu_r: float | complex,
    a: float,
    n: int,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    # TODO: generalize background material
    nn = np.arange(1, n + 1)
    k_d = np.sqrt(eps_r * mu_r) * k_0
    scale_term = (2 * nn + 1) / (nn * (nn + 1))

    den1 = np.sqrt(mu_r) * ric_h2n(nn, k_0 * a) * ric_jnp(nn, k_d * a) - np.sqrt(eps_r) * ric_h2np(
        nn, k_0 * a
    ) * ric_jn(nn, k_d * a)
    den2 = np.sqrt(eps_r) * ric_h2n(nn, k_0 * a) * ric_jnp(nn, k_d * a) - np.sqrt(mu_r) * ric_h2np(
        nn, k_0 * a
    ) * ric_jn(nn, k_d * a)

    an = (
        (1j**-nn)
        * scale_term
        * (
            np.sqrt(eps_r) * ric_jnp(nn, k_0 * a) * ric_jn(nn, k_d * a)
            - np.sqrt(mu_r) * ric_jn(nn, k_0 * a) * ric_jnp(nn, k_d * a)
        )
        / den1
    )
    bn = (
        (1j**-nn)
        * scale_term
        * (
            np.sqrt(mu_r) * ric_jnp(nn, k_0 * a) * ric_jn(nn, k_d * a)
            - np.sqrt(eps_r) * ric_jn(nn, k_0 * a) * ric_jnp(nn, k_d * a)
        )
        / den2
    )
    cn = (1j**-nn) * scale_term * 1j * np.sqrt(eps_r) * mu_r / den1
    dn = (1j**-nn) * scale_term * 1j * np.sqrt(eps_r) * mu_r / den2

    return an, bn, cn, dn


def _calculate_field(
    xyz: npt.NDArray[np.float64],
    k: float | complex,
    eta: float | complex,
    cn: npt.NDArray[np.complex128],
    dn: npt.NDArray[np.complex128],
    fn: Callable,
    dfn: Callable,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    r, theta, phi = scatsol.utils.cart2sph(xyz).T
    nn = np.arange(1, cn.shape[0] + 1)
    theta_cos, phi_cos, phi_sin = np.cos(theta), np.cos(phi), np.sin(phi)

    rbesselterm = fn(nn[:, np.newaxis], k * r)
    rbesselpterm = dfn(nn[:, np.newaxis], k * r)

    p, _ = scatsol.utils.lpmn(1, cn.shape[0], theta_cos)

    p_theta_sin = np.zeros_like(p)
    p_theta_sin[0] = -1
    p_theta_sin[1] = -3 * theta_cos
    for n in range(2, p_theta_sin.shape[0] - 1):
        p_theta_sin[n] = (2 * n + 1) / n * theta_cos * p_theta_sin[n - 1] - (n + 1) / n * p_theta_sin[n - 2]

    dp_theta_sin = np.zeros_like(p)
    dp_theta_sin[0] = theta_cos
    for n in range(2, dp_theta_sin.shape[0]):
        dp_theta_sin[n - 1] = (n + 1) * p_theta_sin[n - 2] - n * np.cos(theta) * p_theta_sin[n - 1]
    dp_theta_sin = -dp_theta_sin

    e_field = np.empty((xyz.shape[0], 3), dtype=np.complex128)
    e_field[:, 0] = ((phi_cos) / (1j * (k * r) ** 2)) * (cn * nn * (nn + 1) @ (rbesselterm * p))
    e_field[:, 1] = -(phi_cos) / (k * r) * (1j * cn @ (rbesselpterm * dp_theta_sin) + dn @ (rbesselterm * p_theta_sin))
    e_field[:, 2] = +(phi_sin) / (k * r) * (1j * cn @ (rbesselpterm * p_theta_sin) + dn @ (rbesselterm * dp_theta_sin))

    h_field = np.empty((xyz.shape[0], 3), dtype=np.complex128)
    h_field[:, 0] = ((phi_sin) / (1j * (k * r) ** 2)) * (dn * nn * (nn + 1) @ (rbesselterm * p))
    h_field[:, 1] = -(phi_sin) / (k * r) * (1j * dn @ (rbesselpterm * dp_theta_sin) + cn @ (rbesselterm * p_theta_sin))
    h_field[:, 2] = -(phi_cos) / (k * r) * (1j * dn @ (rbesselpterm * p_theta_sin) + cn @ (rbesselterm * dp_theta_sin))
    h_field = h_field / eta

    return e_field, h_field
