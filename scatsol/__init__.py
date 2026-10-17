__version__ = "0.2.3"


from scatsol.constant import INFTY, Polarization
from scatsol.geometry import CylindricalGeometry, SphericalGeometry
from scatsol.material import Material
from scatsol.region import CylindricalRegion, SphericalRegion
from scatsol.solver import (
    CylindricalSolution,
    CylindricalSolver,
    SphericalSolution,
    SphericalSolver,
)

__all__ = [
    "INFTY",
    "Polarization",
    "CylindricalGeometry",
    "Material",
    "CylindricalRegion",
    "SphericalGeometry",
    "SphericalRegion",
    "CylindricalSolver",
    "CylindricalSolution",
    "SphericalSolver",
    "SphericalSolution",
]
