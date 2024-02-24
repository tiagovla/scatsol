from scatsol.constant import FREE_SPACE_PERMEABILITY as MU_0
from scatsol.constant import FREE_SPACE_PERMITTIVITY as EPSILON_0
from scatsol.constant import INFTY
from typing import Annotated
import numpy as np


class Material:
    def __init__(self, epsilon_r: float | complex = 1.0, mu_r: float | complex = 1.0) -> None:
        """Representation of a material with electromagnetic properties.

        Args:
            epsilon_r (float | complex): relative electric permittivity.
            mu_r (float | complex): relative magnetic permeability.
        """
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r

    @property
    def epsilon(self) -> float | complex:
        """Calculate the material's real or complex electric permittivity.

        Returns:
            epsilon: electric permittivity.
        """
        return EPSILON_0 * self.epsilon_r

    @property
    def mu(self) -> float | complex:
        """Calculate the material's real or complex magnetic permeability.

        Returns:
            mu: magnetic permeability.
        """
        return MU_0 * self.mu_r

    @property
    def eta(self) -> float | complex:
        """Calculate the material's electromagnetic impedance.

        Returns:
            eta: electromagnetic impedance.
        """
        return np.sqrt(self.mu / self.epsilon)

    def k(self, frequency: Annotated[float, "Hz"]) -> float | complex:
        """Calculate the material's electromagnetic wavenumber.

        Args:
            frequency (float, Hz): frequency.

        Returns:
            k: electromagnetic wavenumber.
        """
        return 2 * np.pi * frequency * np.sqrt(self.mu * self.epsilon)

    def __str__(self) -> str:
        return f"Material(epsilon_r={self.epsilon_r}, mu_r={self.mu_r})"

    def __rep__(self) -> str:
        return self.__str__()

    @classmethod
    def free_space(cls) -> "Material":
        """Create a free space material.

        Returns:
            Material: free space material.
        """
        return cls(epsilon_r=1.0, mu_r=1.0)

    @classmethod
    def PEC(cls) -> "Material":
        """Create a perfect electric conductor material.

        Returns:
            Material: perfect electric conductor material.
        """
        return cls(epsilon_r=-1j*INFTY, mu_r=1.0)

