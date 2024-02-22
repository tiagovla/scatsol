from scatsol.constant import FREE_SPACE_PERMITTIVITY as EPSILON_0
from scatsol.constant import FREE_SPACE_PERMEABILITY as MU_0
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

    def __str__(self) -> str:
        return f"Material(epsilon_r={self.epsilon_r}, mu_r={self.mu_r})"

    def __rep__(self) -> str:
        return self.__str__()
