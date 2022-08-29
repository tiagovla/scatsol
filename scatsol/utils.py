import numpy as np
from scipy.special import hankel2, jv
import numpy.typing as npt
import scipy.special


def jv_prime(v: int, z: float) -> complex:
    """First derivative of the jv function.

    Parameters
    ----------
    v : int
        Order of the function.
    z : float
        Argument of the function.

    Returns
    -------
    complex :
        Evaluated function.

    """
    return 0.5 * (jv(v - 1, z) - jv(v + 1, z))


def hankel2_prime(v: int, z: float) -> complex:
    """First derivative of the hankel2 function.

    Parameters
    ----------
    v : int
        Order of the function.
    z : float
        Argument of the function.

    Returns
    -------
    complex :
        Evaluated function.

    """
    return 0.5 * (hankel2(v - 1, z) - hankel2(v + 1, z))


def generate_position(x1d: np.ndarray, y1d: np.ndarray, z1d: np.ndarray) -> np.ndarray:
    """Generates the position array based on 3 1d arrays.

    Parameters
    ----------
    x1d : np.ndarray
        1d array in the x direction.
    y1d : np.ndarray
        1d array in the y direction.
    z1d : np.ndarray
        1d array in the z direction.

    Returns
    -------
    np.ndarray:
        Position array [Nx3].

    """
    XX, YY, ZZ = np.meshgrid(x1d, y1d, z1d)
    x = np.stack((XX.ravel(), YY.ravel(), ZZ.ravel()), axis=1)
    return x


def cart2spherical(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    rthetaphi = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    rthetaphi[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    rthetaphi[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    rthetaphi[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return rthetaphi


def lpmn(m: int, n: int, x: np.ndarray) -> tuple[npt.NDArray[np.float64], ...]:
    p: npt.NDArray[np.float64] = np.zeros((n, x.shape[0]), dtype=np.float64)
    dp: npt.NDArray[np.float64] = np.zeros_like(p)
    for i in range(x.shape[0]):
        u, du = scipy.special.lpmn(m, n, x[i])
        p[:, i] = u[m, 1:]
        dp[:, i] = du[m, 1:]
    return p, dp
