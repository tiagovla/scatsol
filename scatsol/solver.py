from scatsol.region import CylindricalRegion
from typing import Sequence
from numpy.typing import ArrayLike
from scatsol.constant import Polarization


class CylindricalSolution:
    def __init__(self, n: int, coeficients: Sequence[ArrayLike]):
        self.n = n
        self.coeficients = coeficients

    def scattered_field(self, xyz: ArrayLike):
        pass

    def incident_field(self, xyz: ArrayLike):
        pass


class CylindricalGeometry:
    def __init__(self, regions: Sequence[CylindricalRegion]):
        self.regions = regions
        self._check_limits()

    def _check_limits(self):
        if self.regions[0].inner_radius != 0.0:
            raise ValueError("First region must start at r=0.")
        if self.regions[-1].outer_radius != float("inf"):
            raise ValueError("Last region must end at r=inf.")
        for region_in, region_out in zip(self.regions[:-1], self.regions[1:]):
            if region_in.outer_radius != region_out.inner_radius:
                raise ValueError("Regions limits are not contiguous.")

    def __str__(self) -> str:
        regions = ", ".join([str(region) for region in self.regions])
        return f"CylindricalSolver(regions={regions})"


class CylindricalSolver:
    def __init__(self, geometry: CylindricalGeometry, frequency: float = 300e6):
        self.geometry = geometry
        self.frequency = frequency

    def __str__(self) -> str:
        return f"CylindricalSolver(geometry={self.geometry}, frequency={self.frequency})"

    def solve(self, n: int | None = 50, tolerance: float | None = 1e-6, polarization: Polarization = Polarization.TM):
        if n is None and tolerance is None:
            raise ValueError("n or tolerance must be provided.")
