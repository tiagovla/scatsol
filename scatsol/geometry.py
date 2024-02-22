from typing import Sequence
from scatsol.region import CylindricalRegion


class CylindricalGeometry:
    """Representation of a cylindrical geometry.

    Attributes: 
        regions: sequence of cylindrical regions.
    """
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
