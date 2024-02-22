from scatsol.material import Material


class CylindricalRegion:
    def __init__(self, inner_radius: float, outer_radius: float, material: Material):
        """Cylindrical region representation.

        Args:
            inner_radius: inner radius of the cylindrical region in meters.
            outer_radius: outer radius of the cylindrical region in meters.
            material: material of the cylindrical region.
        """
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.material = material

    def __str__(self) -> str:
        return (
            f"CylindricalRegion(inner_radius={self.inner_radius}, "
            f"outer_radius={self.outer_radius}, material={self.material})"
        )
