import numpy as np
from enum import Enum

PI: float = np.pi
"""Pi constant."""

VACUUM_PERMEABILITY: float = 4 * PI * 1e-7
"""Vacuum permeability [m kg s-2 A-2]."""

SPEED_OF_LIGHT: float = 299792458.0
"""Speed of light in vacuum [m/s]."""

VACUUM_PERMITTIVITY: float = 1 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2)
"""Vacuum permittivity derived from the speed of light and permeability [F/m]."""

INFTY: float = float("inf")
"""Infinity constant."""


class Polarization(Enum):
    TM = "TM"
    TE = "TE"
