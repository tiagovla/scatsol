import numpy as np
import numpy.typing as npt
from scipy.special import spherical_jn as sjn, spherical_yn as syn
from scatsol.material import Material, Medium
import scatsol.utils


def mie_spherical_scattered_field(
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
        an, bn = an_bn_conducting_sphere(bg.k, radius, n)
        Ertp[mask], Hrtp[mask] = calculate_scattered_field_outside(xyz[mask], bg.k, bg.eta, an, bn)
    else:
        s = Medium(sphere, frequency)
        an, bn, cn, dn = an_bn_cn_dn_dielectric_sphere(bg.k, sphere.epsilon_r, sphere.mu_r, radius, n)
        Ertp[mask], Hrtp[mask] = calculate_scattered_field_outside(xyz[mask], bg.k, bg.eta, an, bn)
        Ertp[~mask], Hrtp[~mask] = calculate_scattered_field_inside(xyz[~mask], s.k, s.eta, cn, dn)
    return Ertp, Hrtp


def an_bn_conducting_sphere(k: float, a: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    nn = np.arange(1, n + 1)
    scale_term = (2 * nn + 1) / (nn * (nn + 1))
    sbessel_first = sjn(nn, k * a)
    sbessel_second = syn(nn, k * a)
    sbessel_first_prime = sjn(nn, k * a, derivative=True)
    sbessel_second_prime = syn(nn, k * a, derivative=True)
    bn = -(1j**-nn) * scale_term * sbessel_first / (sbessel_first - 1j * sbessel_second)
    an_num = sbessel_first + k * a * sbessel_first_prime
    an_den = an_num - 1j * (sbessel_second + k * a * sbessel_second_prime)
    an = -(1j**-nn) * scale_term * an_num / an_den
    return an, bn


def an_bn_cn_dn_dielectric_sphere(
    k_0: float, eps_r: float, mu_r: float, a: float, n: int
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

    def ric_jn(z):
        return z * sjn(nn, z)

    def ric_jnp(z):
        return sjn(nn, z) + z * sjn(nn, z, derivative=True)

    def ric_h2(z):
        return z * (sjn(nn, z) - 1j * syn(nn, z))

    def ric_h2p(z):
        return sjn(nn, z) - 1j * syn(nn, z) + z * (sjn(nn, z, derivative=True) - 1j * syn(nn, z, derivative=True))

    den1 = np.sqrt(mu_r) * ric_h2(k_0 * a) * ric_jnp(k_d * a) - np.sqrt(eps_r) * ric_h2p(k_0 * a) * ric_jn(k_d * a)
    den2 = np.sqrt(eps_r) * ric_h2(k_0 * a) * ric_jnp(k_d * a) - np.sqrt(mu_r) * ric_h2p(k_0 * a) * ric_jn(k_d * a)

    an = (
        (1j**-nn)
        * scale_term
        * (np.sqrt(eps_r) * ric_jnp(k_0 * a) * ric_jn(k_d * a) - np.sqrt(mu_r) * ric_jn(k_0 * a) * ric_jnp(k_d * a))
        / den1
    )
    bn = (
        (1j**-nn)
        * scale_term
        * (np.sqrt(mu_r) * ric_jnp(k_0 * a) * ric_jn(k_d * a) - np.sqrt(eps_r) * ric_jn(k_0 * a) * ric_jnp(k_d * a))
        / den2
    )
    cn = (1j**-nn) * scale_term * 1j * np.sqrt(eps_r) * mu_r / den1
    dn = (1j**-nn) * scale_term * 1j * np.sqrt(eps_r) * mu_r / den2

    return an, bn, cn, dn


def calculate_incident_field(xyz, k, n) -> tuple[npt.NDArray[np.complex128], ...]:
    r, theta, _ = scatsol.utils.cart2spherical(xyz).T
    nn = np.arange(1, n + 1)
    sbessel_first = sjn(nn[:, np.newaxis], k * r)
    theta_cos = np.cos(theta)
    p, _ = scatsol.utils.lpmn(0, n, theta_cos)
    e_field = 1j ** (-nn) * (2 * nn + 1) @ (sbessel_first * p)

    return e_field, e_field  # TODO: calculate H field and convert to spherical


def calculate_scattered_field_outside(xyz, k, eta, an, bn) -> tuple[npt.NDArray[np.complex128], ...]:
    r, theta, phi = scatsol.utils.cart2spherical(xyz).T
    nn = np.arange(1, an.shape[0] + 1)

    sj = sjn(nn[:, np.newaxis], k * r)
    sy = syn(nn[:, np.newaxis], k * r)
    sjp = sjn(nn[:, np.newaxis], k * r, derivative=True)
    sjy = syn(nn[:, np.newaxis], k * r, derivative=True)

    theta_cos = np.cos(theta)
    phi_cos = np.cos(phi)
    phi_sin = np.sin(phi)

    ric_h2 = k * r * (sj - 1j * sy)
    ric_h2p = (sj - 1j * sy) + k * r * (sjp - 1j * sjy)
    p, _ = scatsol.utils.lpmn(1, an.shape[0], theta_cos)

    p_theta_sin = np.zeros_like(p)
    p_theta_sin[0] = -1
    p_theta_sin[1] = -3 * np.cos(theta)
    for n in range(2, p_theta_sin.shape[0] - 1):
        p_theta_sin[n] = (2 * n + 1) / n * np.cos(theta) * p_theta_sin[n - 1] - (n + 1) / n * p_theta_sin[n - 2]

    dp_theta_sin = np.zeros_like(p)
    dp_theta_sin[0] = np.cos(theta)
    for n in range(2, dp_theta_sin.shape[0]):
        dp_theta_sin[n - 1] = (n + 1) * p_theta_sin[n - 2] - n * np.cos(theta) * p_theta_sin[n - 1]
    dp_theta_sin = -dp_theta_sin

    e_field = np.empty((xyz.shape[0], 3), dtype=complex)
    e_field[:, 0] = ((phi_cos) / (1j * (k * r) ** 2)) * (an * nn * (nn + 1) @ (ric_h2 * p))
    e_field[:, 1] = -(phi_cos) / (k * r) * (1j * an @ (ric_h2p * dp_theta_sin) + bn @ (ric_h2 * p_theta_sin))
    e_field[:, 2] = +(phi_sin) / (k * r) * (1j * an @ (ric_h2p * p_theta_sin) + bn @ (ric_h2 * dp_theta_sin))

    h_field = np.empty((xyz.shape[0], 3), dtype=complex)
    h_field[:, 0] = ((phi_sin) / (1j * (k * r) ** 2)) * (bn * nn * (nn + 1) @ (ric_h2 * p))
    h_field[:, 1] = -(phi_sin) / (k * r) * (1j * bn @ (ric_h2p * dp_theta_sin) + an @ (ric_h2 * p_theta_sin))
    h_field[:, 2] = -(phi_cos) / (k * r) * (1j * bn @ (ric_h2p * p_theta_sin) + an @ (ric_h2 * dp_theta_sin))
    h_field = h_field / eta

    return e_field, h_field


def calculate_scattered_field_inside(xyz, k, eta, cn, dn) -> tuple[npt.NDArray[np.complex128], ...]:
    r, theta, phi = scatsol.utils.cart2spherical(xyz).T
    nn = np.arange(1, cn.shape[0] + 1)

    sj = sjn(nn[:, np.newaxis], k * r)
    sjp = sjn(nn[:, np.newaxis], k * r, derivative=True)

    theta_cos, phi_cos, phi_sin = np.cos(theta), np.cos(phi), np.sin(phi)

    ric_jv = k * r * (sj)
    ric_jvp = (sj) + k * r * (sjp)
    p, _ = scatsol.utils.lpmn(1, cn.shape[0], theta_cos)

    p_theta_sin = np.zeros_like(p)
    p_theta_sin[0] = -1
    p_theta_sin[1] = -3 * np.cos(theta)
    for n in range(2, p_theta_sin.shape[0] - 1):
        p_theta_sin[n] = (2 * n + 1) / n * np.cos(theta) * p_theta_sin[n - 1] - (n + 1) / n * p_theta_sin[n - 2]

    dp_theta_sin = np.zeros_like(p)
    dp_theta_sin[0] = np.cos(theta)
    for n in range(2, dp_theta_sin.shape[0]):
        dp_theta_sin[n - 1] = (n + 1) * p_theta_sin[n - 2] - n * np.cos(theta) * p_theta_sin[n - 1]
    dp_theta_sin = -dp_theta_sin

    e_field = np.empty((xyz.shape[0], 3), dtype=complex)
    e_field[:, 0] = ((phi_cos) / (1j * (k * r) ** 2)) * (cn * nn * (nn + 1) @ (ric_jv * p))
    e_field[:, 1] = -(phi_cos) / (k * r) * (1j * cn @ (ric_jvp * dp_theta_sin) + dn @ (ric_jv * p_theta_sin))
    e_field[:, 2] = +(phi_sin) / (k * r) * (1j * cn @ (ric_jvp * p_theta_sin) + dn @ (ric_jv * dp_theta_sin))

    h_field = np.empty((xyz.shape[0], 3), dtype=complex)
    h_field[:, 0] = ((phi_sin) / (1j * (k * r) ** 2)) * (dn * nn * (nn + 1) @ (ric_jv * p))
    h_field[:, 1] = -(phi_sin) / (k * r) * (1j * dn @ (ric_jvp * dp_theta_sin) + cn @ (ric_jv * p_theta_sin))
    h_field[:, 2] = -(phi_cos) / (k * r) * (1j * dn @ (ric_jvp * p_theta_sin) + cn @ (ric_jv * dp_theta_sin))
    h_field = h_field / eta

    return e_field, h_field
