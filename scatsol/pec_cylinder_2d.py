import numpy as np
from scipy.special import hankel2, jv

from scatsol.utils import hankel2_prime


class SolutionTM:
    """Represent the TMz solution.

    Parameters
    ----------
    a : float [m]
        Cylinder's radius.
    k : float [1/m]
        Medium's wavelength.
    eta : float [1/m]
        Medium's impedance.
    E0 : float [V/m]
        Amplitude of the incident electric field.

    References
    ----------
    Balanis, Constantine A. Advanced engineering electromagnetics. John Wiley & Sons, 2012.


    """

    def __init__(self, radius: float, k: float, eta: float, E0: float = 1):
        """Represent the TMz solution.

        Parameters
        ----------
        a : float [m]
            Cylinder's radius.
        k : float [1/m]
            Medium's wavelength.
        eta : float [1/m]
            Medium's impedance.
        E0 : float [V/m]
            Amplitude of the incident electric field.
        """
        self.k = k
        self.eta = eta
        self.a = radius
        self.E0 = E0

    def E_inc(self, x: np.ndarray) -> np.ndarray:
        """Calculate the incident electric field.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].

        Returns
        -------
        np.ndarray:
            Numpy array with [Ex, Ey, Ez].

        """
        Ez = self.E0 * np.exp(-1j * self.k * x[:, 0])
        Ex = np.zeros(Ez.shape)
        Ey = np.zeros(Ez.shape)
        return np.stack((Ex, Ey, Ez), axis=1)

    def H_inc(self, x: np.ndarray) -> np.ndarray:
        """Calculate the incident magnetic field.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].

        Returns
        -------
        np.ndarray:
            Numpy array with [Hx, Hy, Hz].

        """
        Hy = -(self.E0 / self.eta) * np.exp(-1j * self.k * x[:, 0])
        Hx = np.zeros(Hy.shape)
        Hz = np.zeros(Hy.shape)
        return np.stack((Hx, Hy, Hz), axis=1)

    def E_scat(self, x: np.ndarray, n: int = 50) -> np.ndarray:
        """Calculate the scat electric field.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].
        n : int
            Number of terms of the modal solution.

        Returns
        -------
        np.ndarray:
            Numpy array with [Ex, Ey, Ez].

        """
        rho = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        phi = np.arctan2(x[:, 1], x[:, 0])

        n = np.arange(n)
        eps = 1.0 + np.heaviside(n - 0.5, 1)
        bessel_term = jv(n, self.k * self.a) / hankel2(n, self.k * self.a)
        cte_term = -self.E0 * eps * (-1j) ** n * bessel_term

        # TODO: use cython + for loop to save memory
        nn, idx = np.meshgrid(n, np.arange(rho.shape[0]), indexing="ij")
        exp_term = hankel2(nn, self.k * rho[idx]) * np.cos(nn * phi[idx])
        Ez = np.einsum("i,ij", cte_term, exp_term)
        Ex = np.zeros(Ez.shape)
        Ey = np.zeros(Ez.shape)
        E = np.stack((Ex, Ey, Ez), axis=1)
        E[rho < self.a] = 0
        return E

    def H_scat(self, x: np.ndarray, n: int = 50) -> np.ndarray:
        """Calculate the scat magnetic field.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].
        n : int
            Number of terms of the modal solution.

        Returns
        -------
        np.ndarray:
            Numpy array with [Hx, Hy, Hz].

        """
        rho = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        phi = np.arctan2(x[:, 1], x[:, 0])

        n = np.arange(n)
        eps = 1.0 + np.heaviside(n - 0.5, 1)
        bessel_term = jv(n, self.k * self.a) / hankel2(n, self.k * self.a)
        cte_term = -self.E0 * eps * (-1j) ** n * bessel_term

        nn, idx = np.meshgrid(n, np.arange(rho.shape[0]), indexing="ij")
        term_1 = (
            self.k
            * np.cos(nn * phi[idx])
            * hankel2_prime(nn, self.k * rho[idx])
            / rho[idx]
        )
        term_2 = (
            nn
            * hankel2(nn, self.k * rho[idx])
            * np.sin(nn * phi[idx])
            / (rho[idx] ** 2)
        )
        exp_term_x = x[:, 0] * term_1 + x[:, 1] * term_2
        exp_term_y = x[:, 1] * term_1 + x[:, 0] * term_2

        Hx = np.einsum("i,ij", cte_term, exp_term_x)
        Hy = np.einsum("i,ij", cte_term, exp_term_y)
        Hz = np.zeros(Hx.shape)
        H = (1j / (self.k * self.eta)) * np.stack((Hx, Hy, Hz), axis=1)
        H[rho < self.a] = 0
        return H

    def E_tot(self, x: np.ndarray, n: int = 50) -> np.ndarray:
        """Calculate the total electric field.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].
        n : int
            Number of terms of the modal solution.

        Returns
        -------
        np.ndarray:
            Numpy array with [Ex, Ey, Ez].

        """
        return self.E_inc(x) + self.E_scat(x, n=n)

    def H_tot(self, x: np.ndarray) -> np.ndarray:
        """Calculate the total magnetic field.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].
        n : int
            Number of terms of the modal solution.

        Returns
        -------
        np.ndarray:
            Numpy array with [Hx, Hy, Hz].

        """
        return self.H_inc(x) + self.H_tot(x)

    def J_eq(self, x: np.ndarray) -> np.ndarray:
        """Calculate the electric current density at the PEC.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].

        """
        raise NotImplementedError

    def M_eq(self, x: np.ndarray) -> np.ndarray:
        """Calculate the magnetic current density at the PEC.

        Parameters
        ----------
        x : np.ndarray
            Numpy array with cartesian positions [Nx3 or Nx2].

        """
        raise NotImplementedError

    def rcs(self, x: np.ndarray) -> np.ndarray:
        """Calculate the RCS."""
        raise NotImplementedError
