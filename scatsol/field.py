import numpy as np
import numpy.typing as npt


class Field:
    def __init__(self, data: npt.NDArray[np.complex128]):
        self.data = data

    def x(self):
        return self.data[:, 0]

    def y(self):
        return self.data[:, 1]

    def z(self):
        return self.data[:, 2]

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
