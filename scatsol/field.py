import numpy as np
import numpy.typing as npt


class Field:
    def __init__(self, field: npt.NDArray[np.complex128]):
        self.field = field

    def x(self):
        return self.field[:, 0]

    def y(self):
        return self.field[:, 1]

    def z(self):
        return self.field[:, 2]

    @classmethod
    def from_cylindrical(
        cls,
        rho: npt.NDArray[np.complex128],
        phi: npt.NDArray[np.complex128],
        z: npt.NDArray[np.complex128],
    ):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return cls(np.array([x, y, z]).T)

    @classmethod
    def from_spherical(
        cls,
        r: npt.NDArray[np.complex128],
        theta: npt.NDArray[np.complex128],
        phi: npt.NDArray[np.complex128],
    ):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return cls(np.array([x, y, z]).T)
