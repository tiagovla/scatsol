import numpy as np
import numpy.typing as npt
import scipy.special
from scipy.special import spherical_jn as sjn, spherical_yn as syn


def field_cyl2cart(field_rtz: npt.NDArray[np.complex128], xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    t_cos, t_sin = np.cos(theta), np.sin(theta)
    field_xyz = np.zeros_like(field_rtz)
    field_xyz[:, 0] = field_rtz[:, 0] * t_cos - field_rtz[:, 1] * t_sin
    field_xyz[:, 1] = field_rtz[:, 0] * t_sin + field_rtz[:, 1] * t_cos
    field_xyz[:, 2] = field_rtz[:, 2]
    return field_xyz


def field_sph2cart(field_rtz: npt.NDArray[np.complex128], xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    _, theta, phi = cart2sph(xyz).T
    t_cos, t_sin = np.cos(theta), np.sin(theta)
    p_cos, p_sin = np.cos(phi), np.sin(phi)
    field_xyz = np.empty_like(field_rtz)
    field_xyz[:, 0] = field_rtz[:, 0] * t_sin * p_cos + field_rtz[:, 1] * t_cos * p_cos - field_rtz[:, 2] * p_sin
    field_xyz[:, 1] = field_rtz[:, 0] * t_sin * p_sin + field_rtz[:, 1] * t_cos * p_sin + field_rtz[:, 2] * p_cos
    field_xyz[:, 2] = field_rtz[:, 0] * t_cos - field_rtz[:, 1] * t_sin
    return field_xyz


def cart2sph(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    rthetaphi = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    rthetaphi[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    rthetaphi[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    rthetaphi[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return rthetaphi


def cart2cyl(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    rphiz = np.zeros(xyz.shape)
    rphiz[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    rphiz[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])
    rphiz[:, 2] = xyz[:, 2]
    return rphiz


def lpmn(m: int, n: int, x: np.ndarray) -> tuple[npt.NDArray[np.float64], ...]:
    p: npt.NDArray[np.float64] = np.zeros((n, x.shape[0]), dtype=np.float64)
    dp: npt.NDArray[np.float64] = np.zeros_like(p)
    for i in range(x.shape[0]):
        u, du = scipy.special.lpmn(m, n, x[i])
        p[:, i] = u[m, 1:]
        dp[:, i] = du[m, 1:]
    return p, dp


def ric_jn(n, z):
    return z * sjn(n, z)


def ric_jnp(n, z):
    return sjn(n, z) + z * sjn(n, z, derivative=True)


def ric_h2n(n, z):
    return z * (sjn(n, z) - 1j * syn(n, z))


def ric_h2np(n, z):
    return sjn(n, z) - 1j * syn(n, z) + z * (sjn(n, z, derivative=True) - 1j * syn(n, z, derivative=True))
