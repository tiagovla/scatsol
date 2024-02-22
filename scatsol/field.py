import numpy as np
from numpy.typing import ArrayLike


class Field:
    def __init__(self, field: np.ndarray):
        self.field = field

    def x(self):
        return self.field[:, 0]

    def y(self):
        return self.field[:, 1]

    def z(self):
        return self.field[:, 2]

    @classmethod
    def from_cylindrical(cls, rho: ArrayLike, phi: ArrayLike, z: ArrayLike):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return cls(np.array([x, y, z]).T)

    @classmethod
    def from_spherical(cls, r: ArrayLike, theta: ArrayLike, phi: ArrayLike):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return cls(np.array([x, y, z]).T)
