"""Gravitational lens equation and ray-tracing mapping.

In gravitational lensing, light rays originating from a source plane position beta
are deflected by an angle alpha(theta) as they pass near a lens, arriving at the observer
from apparent angular position theta in the image plane.

The thin-screen gravitational lens equation relates image plane coordinates theta
to source plane coordinates beta via:
    beta = theta - alpha(theta)

where:
    theta = (theta_x, theta_y) : Image plane coordinate vector.
    beta  = (beta_x, beta_y)   : Source plane coordinate vector.
    alpha = (alpha_x, alpha_y) : Reduced deflection angle vector.

Note on Units:
All quantities (theta, beta, alpha) must share identical angular units (radians or arcseconds).
LensForge enforces strict unit consistency across all ray-tracing operations.
"""

from typing import Tuple, Union
import numpy as np

from lensforge.physics.coordinates import validate_coordinates


def lens_equation(
    theta_x: Union[float, np.ndarray],
    theta_y: Union[float, np.ndarray],
    alpha_x: Union[float, np.ndarray],
    alpha_y: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the gravitational lens equation: beta = theta - alpha(theta).

    Maps image plane positions (theta) to source plane positions (beta) given
    the reduced deflection angle (alpha).

    Args:
        theta_x: Horizontal coordinate(s) in image plane.
        theta_y: Vertical coordinate(s) in image plane.
        alpha_x: Horizontal reduced deflection angle component(s).
        alpha_y: Vertical reduced deflection angle component(s).

    Returns:
        Tuple of (beta_x, beta_y) source plane coordinates with broadcast shape.

    Raises:
        ValueError: If input dimensions cannot be broadcast together.
    """
    tx, ty = validate_coordinates(theta_x, theta_y)
    ax, ay = validate_coordinates(alpha_x, alpha_y)

    try:
        tx_bc, ax_bc = np.broadcast_arrays(tx, ax)
        ty_bc, ay_bc = np.broadcast_arrays(ty, ay)
    except ValueError as exc:
        raise ValueError(
            f"Image coordinates (shape {tx.shape}) and deflection angles (shape {ax.shape}) "
            "cannot be broadcast together."
        ) from exc

    beta_x = tx_bc - ax_bc
    beta_y = ty_bc - ay_bc
    return beta_x, beta_y


def ray_trace_image_to_source(
    theta_x: Union[float, np.ndarray],
    theta_y: Union[float, np.ndarray],
    alpha_x: Union[float, np.ndarray],
    alpha_y: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for lens_equation representing backwards ray-tracing from image to source plane."""
    return lens_equation(theta_x, theta_y, alpha_x, alpha_y)
