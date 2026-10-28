"""Cosmological angular-diameter distances and Einstein radius calculations.

Distances in Gravitational Lensing:
-----------------------------------
A crucial distinction in cosmological gravitational lensing is that all geometric distances
appearing in the lensing equations are ANGULAR-DIAMETER DISTANCES (D_A), not comoving,
luminosity, or Euclidean distances.

The angular-diameter distance is defined by the relation:
    D_A = physical transverse size / angular size = d_phys / delta_theta

Because the Universe expands and spacetime geometry is governed by General Relativity (FLRW metric),
angular-diameter distances are non-additive:
    D_ls != D_s - D_l   (in general cosmological spacetimes)

Only in a static Euclidean spacetime would D_ls = D_s - D_l. In a flat FLRW universe:
    D_ls = [ (1 + z_s) * chi_s - (1 + z_l) * chi_l ] / (1 + z_s)
where chi is the comoving distance.

Therefore, LensForge treats D_l, D_s, and D_ls as distinct physical quantities and strictly validates
their physical boundaries.
"""

from typing import Literal, Union
import numpy as np

from lensforge.constants import C_LIGHT, G_NEWTON, rad2arcsec


def validate_physical_distances(
    mass: float,
    d_l: float,
    d_s: float,
    d_ls: float,
    strict_cosmology: bool = True,
) -> None:
    """Validate physical plausibility of lens mass and angular-diameter distances.

    Args:
        mass: Lens mass in kilograms (must be strictly positive).
        d_l: Observer-to-lens angular-diameter distance in meters (> 0).
        d_s: Observer-to-source angular-diameter distance in meters (> 0).
        d_ls: Lens-to-source angular-diameter distance in meters (> 0).
        strict_cosmology: If True, enforce cosmological ordering (d_s > d_l and d_ls < d_s).

    Raises:
        ValueError: If mass <= 0, distances <= 0, or cosmological ordering is violated.
    """
    if mass <= 0:
        raise ValueError(f"Lens mass must be strictly positive, got {mass} kg.")
    if d_l <= 0:
        raise ValueError(f"Observer-lens distance d_l must be strictly positive, got {d_l} m.")
    if d_s <= 0:
        raise ValueError(f"Observer-source distance d_s must be strictly positive, got {d_s} m.")
    if d_ls <= 0:
        raise ValueError(f"Lens-source distance d_ls must be strictly positive, got {d_ls} m.")

    if strict_cosmology:
        if d_s <= d_l:
            raise ValueError(
                f"Source distance d_s ({d_s:.3e} m) must be strictly greater than lens "
                f"distance d_l ({d_l:.3e} m) in standard gravitational lensing geometry."
            )
        if d_ls >= d_s:
            raise ValueError(
                f"Lens-to-source angular-diameter distance d_ls ({d_ls:.3e} m) cannot exceed "
                f"or equal observer-to-source distance d_s ({d_s:.3e} m)."
            )


def validate_sis_distances(
    velocity_dispersion: float,
    d_s: float,
    d_ls: float,
    strict_cosmology: bool = True,
) -> None:
    """Validate physical plausibility of velocity dispersion and angular-diameter distances for an SIS lens.

    Args:
        velocity_dispersion: 1D velocity dispersion sigma_v in meters/second (> 0).
        d_s: Observer-to-source angular-diameter distance in meters (> 0).
        d_ls: Lens-to-source angular-diameter distance in meters (> 0).
        strict_cosmology: If True, enforce d_ls < d_s.

    Raises:
        ValueError: If sigma_v <= 0, distances <= 0, or d_ls >= d_s.
    """
    if velocity_dispersion <= 0:
        raise ValueError(
            f"Velocity dispersion sigma_v must be strictly positive, got {velocity_dispersion} m/s."
        )
    if d_s <= 0:
        raise ValueError(f"Observer-source distance d_s must be strictly positive, got {d_s} m.")
    if d_ls <= 0:
        raise ValueError(f"Lens-source distance d_ls must be strictly positive, got {d_ls} m.")

    if strict_cosmology and d_ls >= d_s:
        raise ValueError(
            f"Lens-to-source angular-diameter distance d_ls ({d_ls:.3e} m) cannot exceed "
            f"or equal observer-to-source distance d_s ({d_s:.3e} m)."
        )


def einstein_radius(
    mass: float,
    d_l: float,
    d_s: float,
    d_ls: float,
    strict_cosmology: bool = True,
) -> float:
    """Compute the Einstein radius of a point mass in radians.

    The Einstein angular radius is given by the General Relativistic expression:
        theta_E = sqrt( (4 * G * M / c^2) * (D_ls / (D_l * D_s)) )

    where:
        G   : Newtonian gravitational constant (m^3 kg^-1 s^-2)
        c   : Speed of light in vacuum (m s^-1)
        M   : Lens mass (kg)
        D_l : Observer-to-lens angular-diameter distance (m)
        D_s : Observer-to-source angular-diameter distance (m)
        D_ls: Lens-to-source angular-diameter distance (m)

    Args:
        mass: Lens mass in kilograms.
        d_l: Observer-to-lens angular-diameter distance in meters.
        d_s: Observer-to-source angular-diameter distance in meters.
        d_ls: Lens-to-source angular-diameter distance in meters.
        strict_cosmology: If True, validate cosmological distance ordering.

    Returns:
        theta_E: Einstein radius in radians (float).

    Raises:
        ValueError: If input values violate physical bounds.
    """
    validate_physical_distances(mass, d_l, d_s, d_ls, strict_cosmology=strict_cosmology)

    # Schwarzschild radius r_s = 2GM / c^2
    # 4GM / c^2 = 2 * r_s
    schwarzschild_factor = (4.0 * G_NEWTON * mass) / (C_LIGHT**2)
    geometric_ratio = d_ls / (d_l * d_s)
    theta_e_squared = schwarzschild_factor * geometric_ratio

    return float(np.sqrt(theta_e_squared))


def einstein_radius_arcsec(
    mass: float,
    d_l: float,
    d_s: float,
    d_ls: float,
    strict_cosmology: bool = True,
) -> float:
    """Compute the Einstein radius of a point mass in arcseconds.

    Args:
        mass: Lens mass in kilograms.
        d_l: Observer-to-lens angular-diameter distance in meters.
        d_s: Observer-to-source angular-diameter distance in meters.
        d_ls: Lens-to-source angular-diameter distance in meters.
        strict_cosmology: If True, validate cosmological distance ordering.

    Returns:
        theta_E: Einstein radius in arcseconds (float).
    """
    theta_e_rad = einstein_radius(
        mass=mass,
        d_l=d_l,
        d_s=d_s,
        d_ls=d_ls,
        strict_cosmology=strict_cosmology,
    )
    return float(rad2arcsec(theta_e_rad))


def einstein_radius_sis(
    velocity_dispersion: float,
    d_s: float,
    d_ls: float,
    strict_cosmology: bool = True,
) -> float:
    """Compute the Einstein radius of a Singular Isothermal Sphere (SIS) in radians.

    For an isothermal galaxy with 1D velocity dispersion sigma_v:
        theta_E = 4 * pi * (sigma_v / c)^2 * (D_ls / D_s)

    Notice that unlike a point mass, the SIS Einstein radius does not depend on D_l directly,
    only on the distance ratio D_ls / D_s.

    Args:
        velocity_dispersion: 1D stellar velocity dispersion sigma_v in meters/second.
        d_s: Observer-to-source angular-diameter distance in meters.
        d_ls: Lens-to-source angular-diameter distance in meters.
        strict_cosmology: If True, validate cosmological distance ordering (d_ls < d_s).

    Returns:
        theta_E: SIS Einstein radius in radians (float).
    """
    validate_sis_distances(velocity_dispersion, d_s, d_ls, strict_cosmology=strict_cosmology)
    beta_rel = velocity_dispersion / C_LIGHT
    return float(4.0 * np.pi * (beta_rel**2) * (d_ls / d_s))


def einstein_radius_sis_arcsec(
    velocity_dispersion: float,
    d_s: float,
    d_ls: float,
    strict_cosmology: bool = True,
) -> float:
    """Compute the Einstein radius of an SIS galaxy in arcseconds.

    Args:
        velocity_dispersion: 1D velocity dispersion sigma_v in meters/second.
        d_s: Observer-to-source angular-diameter distance in meters.
        d_ls: Lens-to-source angular-diameter distance in meters.
        strict_cosmology: If True, validate cosmological distance ordering.

    Returns:
        theta_E: SIS Einstein radius in arcseconds (float).
    """
    theta_e_rad = einstein_radius_sis(
        velocity_dispersion=velocity_dispersion,
        d_s=d_s,
        d_ls=d_ls,
        strict_cosmology=strict_cosmology,
    )
    return float(rad2arcsec(theta_e_rad))
