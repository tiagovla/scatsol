from scatsol.constant import FREE_SPACE_PERMITTIVITY as EPSILON_0
from scatsol.constant import FREE_SPACE_PERMEABILITY as MU_0
import numpy as np


class Material:
    def __init__(self, epsilon_r: float = 1.0, mu_r: float = 1.0) -> None:
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r

    @property
    def epsilon(self) -> float:
        return EPSILON_0 * self.epsilon_r

    @property
    def mu(self) -> float:
        return MU_0 * self.mu_r

    @property
    def eta(self) -> float:
        return np.sqrt(self.mu / self.epsilon)

    def __str__(self) -> str:
        return f"Material(epsilon_r={self.epsilon_r}, mu_r={self.mu_r})"

    def __rep__(self) -> str:
        return self.__str__()


class Medium:
    def __init__(self, material: Material, frequency: float = 300e6) -> None:
        self.material = material
        self.frequency = frequency

    @property
    def k(self) -> float:
        return (
            2.0
            * np.pi
            * self.frequency
            * np.sqrt(self.material.epsilon * self.material.mu)
        )

    @property
    def eta(self) -> float:
        return np.sqrt(self.material.mu / self.material.epsilon)

    @property
    def omega(self) -> float:
        return np.sqrt(self.material.mu / self.material.epsilon)
