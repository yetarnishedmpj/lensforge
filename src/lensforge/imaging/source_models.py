"""Astrophysical background source models and ray-traced surface-brightness sampling.

Liouville's Theorem & Conservation of Surface Brightness:
---------------------------------------------------------
Gravitational deflection is achromatic and conserves phase-space density.
Consequently, gravitational lensing conserves surface brightness:
    I_obs(theta) = S(beta(theta))

where:
    I_obs(theta) : Observed surface brightness at image plane coordinate theta.
    S(beta)      : Intrinsic surface brightness of the source galaxy at source plane coordinate beta.
    beta(theta)  : Source position connected to theta via the lens equation beta = theta - alpha(theta).

In this preliminary phase, we perform ideal backwards ray-tracing directly on continuous
analytic source distributions S(beta). Telescope point-spread function (PSF) convolution
and observational noise (N) will be incorporated in subsequent phases.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np

from lensforge.physics.coordinates import CoordinateGrid, validate_coordinates


class BaseSourceModel(ABC):
    """Abstract base class for 2D astronomical source surface-brightness distributions."""

    @abstractmethod
    def evaluate(
        self,
        beta_x: Union[float, np.ndarray],
        beta_y: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Evaluate intrinsic surface brightness S(beta_x, beta_y).

        Args:
            beta_x: Horizontal coordinate(s) in the source plane.
            beta_y: Vertical coordinate(s) in the source plane.

        Returns:
            Surface brightness array in arbitrary or physical flux units (e.g., Jy / arcsec^2).
        """
        pass


class GaussianSource(BaseSourceModel):
    """Two-dimensional Gaussian surface-brightness source profile.

    S(beta_x, beta_y) = I_0 * exp( -0.5 * ( (dx_rot / sigma_x)^2 + (dy_rot / sigma_y)^2 ) )

    Attributes:
        center: (x_0, y_0) source centroid.
        sigma: Circular dispersion scale (or semi-major axis scale if axis_ratio != 1.0).
        amp: Peak central surface brightness I_0.
        axis_ratio: Minor-to-major axis ratio q (0 < q <= 1). Default 1.0 (circular).
        pos_angle: Position angle in radians, measured counterclockwise from the positive x-axis.
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 0.2,
        amp: float = 1.0,
        axis_ratio: float = 1.0,
        pos_angle: float = 0.0,
    ) -> None:
        """Initialize the 2D Gaussian source model.

        Args:
            center: (x_0, y_0) source centroid in angular units.
            sigma: Standard deviation scale (> 0).
            amp: Central peak surface brightness (> 0).
            axis_ratio: Axis ratio q = b/a (0 < q <= 1).
            pos_angle: Position angle phi in radians.

        Raises:
            ValueError: If sigma <= 0, amp <= 0, or axis_ratio not in (0, 1].
        """
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}.")
        if amp <= 0.0:
            raise ValueError(f"amp must be strictly positive, got {amp}.")
        if not (0.0 < axis_ratio <= 1.0):
            raise ValueError(f"axis_ratio must satisfy 0 < axis_ratio <= 1, got {axis_ratio}.")

        self.center = (float(center[0]), float(center[1]))
        self.sigma = float(sigma)
        self.amp = float(amp)
        self.axis_ratio = float(axis_ratio)
        self.pos_angle = float(pos_angle)

    @classmethod
    def from_degrees(
        cls,
        center: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 0.2,
        amp: float = 1.0,
        axis_ratio: float = 1.0,
        pos_angle_deg: float = 0.0,
    ) -> "GaussianSource":
        """Instantiate a Gaussian source with position angle in degrees."""
        return cls(
            center=center,
            sigma=sigma,
            amp=amp,
            axis_ratio=axis_ratio,
            pos_angle=float(np.radians(pos_angle_deg)),
        )

    @property
    def fwhm(self) -> float:
        """Full Width at Half Maximum (FWHM) of the profile."""
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma

    @property
    def total_flux(self) -> float:
        """Total analytically integrated flux: 2 * pi * amp * sigma^2 * axis_ratio."""
        return 2.0 * np.pi * self.amp * (self.sigma**2) * self.axis_ratio

    def evaluate(
        self,
        beta_x: Union[float, np.ndarray],
        beta_y: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Evaluate the Gaussian surface brightness at coordinates (beta_x, beta_y).

        Args:
            beta_x: Horizontal coordinate(s).
            beta_y: Vertical coordinate(s).

        Returns:
            Surface brightness array.
        """
        bx, by = validate_coordinates(beta_x, beta_y)
        dx = bx - self.center[0]
        dy = by - self.center[1]

        if self.pos_angle != 0.0:
            cos_phi = np.cos(self.pos_angle)
            sin_phi = np.sin(self.pos_angle)
            dx_rot = dx * cos_phi + dy * sin_phi
            dy_rot = -dx * sin_phi + dy * cos_phi
        else:
            dx_rot = dx
            dy_rot = dy

        sigma_x = self.sigma
        sigma_y = self.sigma * self.axis_ratio

        exponent = -0.5 * ((dx_rot / sigma_x) ** 2 + (dy_rot / sigma_y) ** 2)
        return self.amp * np.exp(exponent)


def simulate_lensed_image(
    lens_model,
    source_model: BaseSourceModel,
    grid: CoordinateGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate ideal gravitational lensing via backwards ray-tracing on a coordinate grid.

    Calculates:
        1. Source brightness without lensing: S(theta)
        2. Lensed image brightness via Liouville's theorem: I_obs(theta) = S(beta(theta))

    Args:
        lens_model: Lens instance supporting ray_trace(theta_x, theta_y, mask_singular=True).
        source_model: BaseSourceModel instance representing the background galaxy.
        grid: CoordinateGrid defining the image plane pixel sampling.

    Returns:
        Tuple of (unlensed_image, lensed_image) 2D numpy arrays.
    """
    # 1. Unlensed surface brightness on the coordinate grid
    unlensed_image = source_model.evaluate(grid.x, grid.y)

    # 2. Deflection and lens equation ray-tracing to find source coordinates
    beta_x, beta_y = lens_model.ray_trace(grid.x, grid.y, mask_singular=True)

    # 3. Sample source at ray-traced coordinates
    # Masked pixels (singularities) receive 0 flux
    valid_mask = np.isfinite(beta_x) & np.isfinite(beta_y)
    lensed_image = np.zeros_like(grid.x, dtype=np.float64)

    if np.any(valid_mask):
        lensed_image[valid_mask] = source_model.evaluate(
            beta_x[valid_mask],
            beta_y[valid_mask],
        )

    return unlensed_image, lensed_image
