"""Gravitational lensing magnification and parity for point-mass lenses.

Magnification in Gravitational Lensing:
---------------------------------------
The magnification factor mu of a lensed image is determined by the inverse determinant
of the Jacobian matrix A of the lens mapping:
    A_ij = d beta_i / d theta_j = delta_ij - d^2 psi / (d theta_i d theta_j)
    mu = 1 / det(A)

Physical Interpretation of Signed Magnification & Parity:
---------------------------------------------------------
- Positive magnification (mu > 0, Parity = +1):
  The image maintains the same spatial orientation (handedness) as the source.
  For a point-mass lens, this is the outer image (theta_+), where |theta_+| > theta_E.
- Negative magnification (mu < 0, Parity = -1):
  The image is mirrored / inverted across the critical curve.
  For a point-mass lens, this is the inner image (theta_-), where |theta_-| < theta_E.
- OBSERVED FLUX IS ALWAYS POSITIVE:
  Telescopes measure energy flux F_obs = |mu| * F_source. The negative sign is a mathematical
  topological property denoting orientation reversal (parity), NEVER negative radiant energy!
- Magnification Invariant:
  For any point-mass lens, mu_+ + mu_- = 1 identically for all u > 0.

Caustic & Singularity:
----------------------
At beta = 0 (perfect alignment, u = 0), the source lies exactly on the point caustic.
The formal point-source magnification diverges to infinity (mu -> inf).
"""

from typing import Tuple, Union
import numpy as np

ArrayLike = Union[float, np.ndarray]


def point_mass_magnification(
    u: ArrayLike,
    allow_infinite: bool = False,
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculate signed image magnifications (mu_plus, mu_minus) for normalized source offset u.

    For normalized source impact parameter u = |beta| / theta_E > 0:
        mu_plus  = 1/2 + (u^2 + 2) / (2 * u * sqrt(u^2 + 4))
        mu_minus = 1/2 - (u^2 + 2) / (2 * u * sqrt(u^2 + 4))

    Args:
        u: Normalized angular offset u = |beta| / theta_E (scalar or numpy array). Must be >= 0.
        allow_infinite: If True, returns (inf, -inf) when u == 0.
                        If False (default), raises ZeroDivisionError when u == 0.

    Returns:
        Tuple of (mu_plus, mu_minus) signed magnifications matching input shape.
        mu_plus is positive (> 1.0).
        mu_minus is negative (< 0.0), representing parity reversal.

    Raises:
        ValueError: If any u < 0.
        ZeroDivisionError: If any u == 0 and allow_infinite is False.
    """
    u_arr = np.asarray(u, dtype=np.float64)

    if np.any(u_arr < 0.0):
        raise ValueError(f"Normalized source offset u must be non-negative, got minimum value {np.min(u_arr)}.")

    has_zero = np.any(u_arr == 0.0)
    if has_zero and not allow_infinite:
        raise ZeroDivisionError(
            "Point-mass magnification diverges to infinity at u = 0 (source directly on the caustic). "
            "Pass allow_infinite=True to allow returning infinity."
        )

    # Compute safely using numerically stable rationalized form:
    # Notice that 1/2 - (u^2 + 2) / (2u * sqrt(u^2 + 4)) suffers from catastrophic cancellation
    # as u -> infinity, since (u^2 + 2) / (2u * sqrt(u^2 + 4)) -> 1/2.
    # Rationalizing the numerator yields:
    #   mu_minus = -2 / (u * sqrt(u^2 + 4) * (u * sqrt(u^2 + 4) + u^2 + 2))
    # which is algebraically exact for all u > 0, completely immune to cancellation,
    # and strictly negative.
    # By the fundamental invariant mu_+ + mu_- = 1, mu_+ = 1 - mu_minus.
    with np.errstate(divide="ignore", invalid="ignore"):
        u_sq = u_arr**2
        sqrt_term = np.sqrt(u_sq + 4.0)
        u_sqrt = u_arr * sqrt_term
        denom_minus = u_sqrt * (u_sqrt + u_sq + 2.0)
        mu_minus = -2.0 / denom_minus
        mu_plus = 1.0 - mu_minus

        if has_zero:
            mu_plus = np.where(u_arr == 0.0, np.inf, mu_plus)
            mu_minus = np.where(u_arr == 0.0, -np.inf, mu_minus)

    if np.isscalar(u) or (isinstance(u, (float, int))):
        return float(mu_plus), float(mu_minus)
    return mu_plus, mu_minus


def point_mass_absolute_magnification(
    u: ArrayLike,
    allow_infinite: bool = False,
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculate absolute observed image magnifications (|mu_plus|, |mu_minus|).

    Observed image flux is proportional to the absolute magnification:
        flux_+ = |mu_+| * flux_source
        flux_- = |mu_-| * flux_source

    Args:
        u: Normalized angular offset u = |beta| / theta_E (must be >= 0).
        allow_infinite: Whether to allow infinite magnification at u = 0.

    Returns:
        Tuple of (|mu_plus|, |mu_minus|), both strictly positive for u > 0.
    """
    mu_plus, mu_minus = point_mass_magnification(u, allow_infinite=allow_infinite)
    if isinstance(mu_plus, float):
        return abs(mu_plus), abs(mu_minus)
    return np.abs(mu_plus), np.abs(mu_minus)


def point_mass_total_magnification(
    u: ArrayLike,
    allow_infinite: bool = False,
) -> ArrayLike:
    """Calculate the total unresolved flux magnification for a point source.

    The total magnification is the sum of the absolute flux magnifications of both images:
        mu_tot = |mu_+| + |mu_-| = mu_+ - mu_- = (u^2 + 2) / (u * sqrt(u^2 + 4))

    Properties:
        - As u -> 0: mu_tot -> infinity (caustic divergence).
        - As u -> infinity: mu_tot -> 1 (unlensed limit).
        - For all finite u > 0: mu_tot > 1 (lensing always amplifies total flux for a point mass).

    Args:
        u: Normalized angular offset u = |beta| / theta_E.
        allow_infinite: Whether to return inf when u == 0.

    Returns:
        Total magnification mu_tot (scalar or array).
    """
    abs_plus, abs_minus = point_mass_absolute_magnification(u, allow_infinite=allow_infinite)
    return abs_plus + abs_minus
