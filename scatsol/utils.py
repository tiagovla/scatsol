import numpy as np
from scipy.special import hankel2, jv


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
