"""Effective gravitational lens potential and numerical gradient utilities.

Lensing Potential:
------------------
The effective 2D gravitational lens potential psi(theta) is related to the deflection angle
via the 2D gradient on the celestial sphere:
    alpha(theta) = grad_theta psi(theta) = ( d psi / d theta_x, d psi / d theta_y )

For a point-mass lens centered at theta_l with Einstein radius theta_E:
    psi(theta) = theta_E^2 * ln( |theta - theta_l| / theta_ref )

where theta_ref is an arbitrary positive reference angle.
Since ln(|theta - theta_l| / theta_ref) = ln|theta - theta_l| - ln(theta_ref), the choice of
theta_ref simply adds an unobservable global constant -theta_E^2 * ln(theta_ref) to the potential,
leaving the deflection field grad psi completely invariant:
    d psi / d theta_x = theta_E^2 * (theta_x - theta_l_x) / |theta - theta_l|^2 = alpha_x
    d psi / d theta_y = theta_E^2 * (theta_y - theta_l_y) / |theta - theta_l|^2 = alpha_y

Singularity:
------------
At theta = theta_l, |theta - theta_l| = 0, causing a logarithmic divergence ln(0) -> -inf.
This singularity must be handled explicitly.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np

from lensforge.physics.coordinates import validate_coordinates


class SingularityError(ValueError):
    """Exception raised when evaluating lensing potential or deflection at a singularity."""


def point_mass_potential(
    theta_x: Union[float, np.ndarray],
    theta_y: Union[float, np.ndarray],
    theta_e: float,
    center: Tuple[float, float] = (0.0, 0.0),
    theta_ref: float = 1.0,
    mask_singular: bool = False,
) -> np.ndarray:
    """Compute the 2D effective gravitational potential for a point-mass lens.

    psi(theta) = theta_E^2 * ln( |theta - center| / theta_ref )

    Args:
        theta_x: Horizontal coordinate(s).
        theta_y: Vertical coordinate(s).
        theta_e: Einstein radius (> 0) in the same angular units as theta.
        center: (center_x, center_y) coordinates of the lens mass.
        theta_ref: Positive reference scale angle (> 0). Default is 1.0.
        mask_singular: If False (default), raises SingularityError when |theta - center| == 0.
                       If True, suppresses error and places -np.inf at singular positions.

    Returns:
        Potential psi(theta) as numpy array matching broadcast coordinate shape.

    Raises:
        ValueError: If theta_e <= 0 or theta_ref <= 0.
        SingularityError: If any evaluation point coincides with lens center and mask_singular is False.
    """
    if theta_e <= 0:
        raise ValueError(f"Einstein radius theta_e must be strictly positive, got {theta_e}.")
    if theta_ref <= 0:
        raise ValueError(f"Reference scale theta_ref must be strictly positive, got {theta_ref}.")

    tx, ty = validate_coordinates(theta_x, theta_y)
    dx = tx - center[0]
    dy = ty - center[1]
    r = np.hypot(dx, dy)

    is_singular = (r == 0.0)
    if np.any(is_singular):
        if not mask_singular:
            raise SingularityError(
                "Evaluation coordinate coincides exactly with point-mass lens singularity at "
                f"center {center}. Use mask_singular=True to handle singular points."
            )

    # Compute potential safely with np.errstate to handle singular points if mask_singular is True
    with np.errstate(divide="ignore"):
        normalized_r = r / theta_ref
        psi = (theta_e**2) * np.log(normalized_r)

    return psi


def gradient_finite_difference(
    potential_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_x: Union[float, np.ndarray],
    theta_y: Union[float, np.ndarray],
    step_size: float = 1.0e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the numerical gradient of a 2D scalar potential using central finite differences.

    grad(psi) = ( (psi(x+h, y) - psi(x-h, y)) / (2h), (psi(x, y+h) - psi(x, y-h)) / (2h) )

    Args:
        potential_func: Callable taking (x, y) arrays and returning potential array.
        theta_x: Horizontal coordinate(s).
        theta_y: Vertical coordinate(s).
        step_size: Finite difference step h (> 0). Default 1e-6.

    Returns:
        Tuple of (d_psi_dx, d_psi_dy) numpy arrays representing numerical gradient.

    Raises:
        ValueError: If step_size <= 0.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be strictly positive, got {step_size}.")

    tx, ty = validate_coordinates(theta_x, theta_y)
    h = step_size

    psi_x_plus = potential_func(tx + h, ty)
    psi_x_minus = potential_func(tx - h, ty)
    grad_x = (psi_x_plus - psi_x_minus) / (2.0 * h)

    psi_y_plus = potential_func(tx, ty + h)
    psi_y_minus = potential_func(tx, ty - h)
    grad_y = (psi_y_plus - psi_y_minus) / (2.0 * h)

    return grad_x, grad_y
