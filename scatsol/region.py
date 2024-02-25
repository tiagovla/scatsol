from scatsol.material import Material
from typing import Annotated


class CylindricalRegion:
    def __init__(
        self,
        inner_radius: Annotated[float, "m"],
        outer_radius: Annotated[float, "m"],
        material: Material,
    ):
        """Cylindrical region representation.

        Args:
            inner_radius: inner radius of the cylindrical region in meters.
            outer_radius: outer radius of the cylindrical region in meters.
            material: material of the cylindrical region.
        """
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.material = material

    def check_within(self, radius: Annotated[float, "m"]) -> bool:
        """Check if a radius is within the region not including the outer radius.

        Args:
            radius: radius in meters to check.

        Returns:
            True if the radius is within the region, False otherwise.
        """
        return self.inner_radius <= radius < self.outer_radius

    def __str__(self) -> str:
        return (
            f"CylindricalRegion(inner_radius={self.inner_radius}, "
            f"outer_radius={self.outer_radius}, material={self.material})"
        )
