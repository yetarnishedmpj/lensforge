"""Physical constants and unit conversion utilities for LensForge.

All physical calculations in LensForge use the International System of Units (SI)
internally unless explicitly stated otherwise in docstrings.

Constants:
    C_LIGHT: Speed of light in vacuum (m/s) [CODATA 2018/2022 exact value].
    G_NEWTON: Newtonian gravitational constant (m^3 kg^-1 s^-2) [CODATA 2018].
    M_SUN: Solar mass (kg) [IAU 2015 nominal solar mass parameter / G].
    PARSEC: 1 parsec in meters (m) [IAU definition: 648,000 / pi AU].
    MEGAPARSEC: 1 megaparsec in meters (m).

Angular Conversion Factors:
    ARCSEC_PER_RAD: Number of arcseconds per radian (648000 / pi).
    RAD_PER_ARCSEC: Number of radians per arcsecond (pi / 648000).
    DEG_PER_RAD: Number of degrees per radian (180 / pi).
    RAD_PER_DEG: Number of radians per degree (pi / 180).
"""

from typing import Union
import numpy as np

# Physical Constants (SI Units)
# Speed of light in vacuum (exact SI definition) [m / s]
C_LIGHT: float = 299_792_458.0

# Newtonian gravitational constant (CODATA 2018 recommended value) [m^3 / (kg s^2)]
G_NEWTON: float = 6.67430e-11

# Solar mass (standard astrophysical reference) [kg]
# Based on IAU 2015 Resolution B3: nominal solar conversion (GM_sun = 1.3271244e20 m^3/s^2)
M_SUN: float = 1.98847e30

# Distance conversion constants [m]
# 1 astronomical unit (exact definition) [m]
ASTRONOMICAL_UNIT: float = 149_597_870_700.0

# 1 parsec = (648000 / pi) * AU [m]
PARSEC: float = (648_000.0 / np.pi) * ASTRONOMICAL_UNIT  # ~ 3.08567758149e16 m

# 1 megaparsec = 10^6 parsecs [m]
MEGAPARSEC: float = 1.0e6 * PARSEC  # ~ 3.08567758149e22 m

# Angular Conversion Factors
ARCSEC_PER_RAD: float = 648_000.0 / np.pi
RAD_PER_ARCSEC: float = np.pi / 648_000.0
DEG_PER_RAD: float = 180.0 / np.pi
RAD_PER_DEG: float = np.pi / 180.0
ARCSEC_PER_DEG: float = 3_600.0
DEG_PER_ARCSEC: float = 1.0 / 3_600.0

# Type alias for float or numpy array
ArrayLike = Union[float, int, np.ndarray]


def rad2arcsec(rad: ArrayLike) -> ArrayLike:
    """Convert an angle from radians to arcseconds.

    Args:
        rad: Angle in radians (scalar float, int, or numpy ndarray).

    Returns:
        Angle in arcseconds with the same shape/type as input.
    """
    return np.asarray(rad) * ARCSEC_PER_RAD if isinstance(rad, (list, tuple)) else rad * ARCSEC_PER_RAD


def arcsec2rad(arcsec: ArrayLike) -> ArrayLike:
    """Convert an angle from arcseconds to radians.

    Args:
        arcsec: Angle in arcseconds (scalar float, int, or numpy ndarray).

    Returns:
        Angle in radians with the same shape/type as input.
    """
    return np.asarray(arcsec) * RAD_PER_ARCSEC if isinstance(arcsec, (list, tuple)) else arcsec * RAD_PER_ARCSEC


def deg2rad(deg: ArrayLike) -> ArrayLike:
    """Convert an angle from degrees to radians.

    Args:
        deg: Angle in degrees (scalar float, int, or numpy ndarray).

    Returns:
        Angle in radians with the same shape/type as input.
    """
    return np.asarray(deg) * RAD_PER_DEG if isinstance(deg, (list, tuple)) else deg * RAD_PER_DEG


def rad2deg(rad: ArrayLike) -> ArrayLike:
    """Convert an angle from radians to degrees.

    Args:
        rad: Angle in radians (scalar float, int, or numpy ndarray).

    Returns:
        Angle in degrees with the same shape/type as input.
    """
    return np.asarray(rad) * DEG_PER_RAD if isinstance(rad, (list, tuple)) else rad * DEG_PER_RAD


def arcsec2deg(arcsec: ArrayLike) -> ArrayLike:
    """Convert an angle from arcseconds to degrees.

    Args:
        arcsec: Angle in arcseconds (scalar float, int, or numpy ndarray).

    Returns:
        Angle in degrees with the same shape/type as input.
    """
    return np.asarray(arcsec) * DEG_PER_ARCSEC if isinstance(arcsec, (list, tuple)) else arcsec * DEG_PER_ARCSEC


def deg2arcsec(deg: ArrayLike) -> ArrayLike:
    """Convert an angle from degrees to arcseconds.

    Args:
        deg: Angle in degrees (scalar float, int, or numpy ndarray).

    Returns:
        Angle in arcseconds with the same shape/type as input.
    """
    return np.asarray(deg) * ARCSEC_PER_DEG if isinstance(deg, (list, tuple)) else deg * ARCSEC_PER_DEG
