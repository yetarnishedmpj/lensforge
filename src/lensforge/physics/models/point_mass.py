"""Schwarzschild point-mass gravitational lens model.

Mathematical Framework:
-----------------------
A point-mass lens is the fundamental analytic model in gravitational lensing.
For a point mass with Einstein radius theta_E centered at theta_l:

1. Reduced Deflection Angle:
   alpha(theta) = theta_E^2 * (theta - theta_l) / |theta - theta_l|^2

2. Effective Lensing Potential:
   psi(theta) = theta_E^2 * ln( |theta - theta_l| / theta_ref )

3. Lens Equation (Ray Tracing):
   beta = theta - alpha(theta)

4. Image Positions for Point Source:
   Scalar 1D collinear offset beta along axis from lens:
       theta_+ = (beta + sqrt(beta^2 + 4 * theta_E^2)) / 2
       theta_- = (beta - sqrt(beta^2 + 4 * theta_E^2)) / 2

   Special Case (Perfect Alignment, beta = 0):
   When beta = 0, rotational symmetry dictates that the source is mapped to a continuous
   circular Einstein ring of radius theta_E, rather than two isolated image points.

5. Magnification:
   For normalized source offset u = |beta - theta_l| / theta_E:
       mu_+ = 1/2 + (u^2 + 2) / (2u * sqrt(u^2 + 4))   (outer image, positive parity)
       mu_- = 1/2 - (u^2 + 2) / (2u * sqrt(u^2 + 4))   (inner image, negative parity)
"""

from typing import Optional, Tuple, Union
import numpy as np

from lensforge.constants import arcsec2rad, rad2arcsec
from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.distances import einstein_radius
from lensforge.physics.lens_equation import lens_equation
from lensforge.physics.lens_potential import SingularityError, point_mass_potential
from lensforge.physics.magnification import (
    point_mass_absolute_magnification,
    point_mass_magnification,
    point_mass_total_magnification,
)
from lensforge.physics.models.base import BaseLensModel


class PerfectAlignmentWarning(UserWarning):
    """Warning emitted when computing point-image positions under perfect alignment (Einstein ring)."""


class PointMassLens(BaseLensModel):
    """Analytical Schwarzschild point-mass gravitational lens model.

    Attributes:
        theta_e: Einstein radius in radians (or consistent angular units).
        center: (center_x, center_y) coordinates of the lens center.
    """

    def __init__(
        self,
        theta_e: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize the point-mass lens model.

        Args:
            theta_e: Einstein radius in radians (must be strictly positive).
            center: (center_x, center_y) coordinates of the lens center in radians.

        Raises:
            ValueError: If theta_e <= 0.
        """
        if theta_e <= 0.0:
            raise ValueError(f"Einstein radius theta_e must be strictly positive, got {theta_e}.")

        self._theta_e = float(theta_e)
        self._center = (float(center[0]), float(center[1]))

    @classmethod
    def from_physical_parameters(
        cls,
        mass: float,
        d_l: float,
        d_s: float,
        d_ls: float,
        center: Tuple[float, float] = (0.0, 0.0),
        strict_cosmology: bool = True,
    ) -> "PointMassLens":
        """Instantiate a point-mass lens from physical mass and angular-diameter distances.

        Args:
            mass: Lens mass in kilograms (> 0).
            d_l: Observer-to-lens angular-diameter distance in meters (> 0).
            d_s: Observer-to-source angular-diameter distance in meters (> 0).
            d_ls: Lens-to-source angular-diameter distance in meters (> 0).
            center: (center_x, center_y) lens center in radians.
            strict_cosmology: If True, enforce cosmological distance bounds.

        Returns:
            PointMassLens configured with calculated Einstein radius in radians.
        """
        theta_e = einstein_radius(
            mass=mass,
            d_l=d_l,
            d_s=d_s,
            d_ls=d_ls,
            strict_cosmology=strict_cosmology,
        )
        return cls(theta_e=theta_e, center=center)

    @property
    def theta_e(self) -> float:
        """Einstein radius of the lens."""
        return self._theta_e

    @property
    def theta_e_arcsec(self) -> float:
        """Einstein radius converted to arcseconds."""
        return float(rad2arcsec(self._theta_e))

    @property
    def center(self) -> Tuple[float, float]:
        """(center_x, center_y) coordinates of the lens."""
        return self._center

    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the reduced deflection field alpha(theta).

        alpha(theta) = theta_E^2 * (theta - theta_l) / |theta - theta_l|^2

        Args:
            theta_x: Horizontal coordinate(s) in image plane.
            theta_y: Vertical coordinate(s) in image plane.
            mask_singular: If False (default), raise SingularityError if any coordinate
                           lies exactly at the lens center.
                           If True, sets deflection to NaN at the singular point.

        Returns:
            Tuple of (alpha_x, alpha_y) deflection angles as numpy arrays.

        Raises:
            SingularityError: If theta hits the lens center and mask_singular is False.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        r_squared = dx**2 + dy**2

        is_singular = (r_squared == 0.0)
        if np.any(is_singular):
            if not mask_singular:
                raise SingularityError(
                    f"Deflection field is singular at lens center {self._center}. "
                    "Set mask_singular=True to return NaN at singular coordinates."
                )

        with np.errstate(divide="ignore", invalid="ignore"):
            factor = (self._theta_e**2) / r_squared
            alpha_x = factor * dx
            alpha_y = factor * dy

            if np.any(is_singular):
                alpha_x = np.where(is_singular, np.nan, alpha_x)
                alpha_y = np.where(is_singular, np.nan, alpha_y)

        return alpha_x, alpha_y

    def potential(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        theta_ref: Optional[float] = None,
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the effective 2D gravitational lens potential psi(theta).

        psi(theta) = theta_E^2 * ln( |theta - theta_l| / theta_ref )

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            theta_ref: Reference angular scale (default is theta_e).
            mask_singular: If False, raises SingularityError at lens center.

        Returns:
            Lensing potential array.
        """
        ref = self._theta_e if theta_ref is None else theta_ref
        return point_mass_potential(
            theta_x=theta_x,
            theta_y=theta_y,
            theta_e=self._theta_e,
            center=self._center,
            theta_ref=ref,
            mask_singular=mask_singular,
        )

    def ray_trace(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map image plane coordinates theta back to source plane coordinates beta.

        beta = theta - alpha(theta)

        Args:
            theta_x: Horizontal image coordinates.
            theta_y: Vertical image coordinates.
            mask_singular: Singularity masking policy passed to deflection.

        Returns:
            Tuple of (beta_x, beta_y) source coordinates.
        """
        alpha_x, alpha_y = self.deflection(theta_x, theta_y, mask_singular=mask_singular)
        return lens_equation(theta_x, theta_y, alpha_x, alpha_y)

    @staticmethod
    def scalar_image_positions(
        beta: Union[float, np.ndarray],
        theta_e: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the two analytical collinear image positions for a scalar source offset beta.

        Given scalar source position beta and Einstein radius theta_E:
            theta_+ = (beta + sqrt(beta^2 + 4 * theta_E^2)) / 2  (outer image, > 0 for beta >= 0)
            theta_- = (beta - sqrt(beta^2 + 4 * theta_E^2)) / 2  (inner image, < 0 for beta >= 0)

        Special Case (beta = 0):
            theta_+ = +theta_E
            theta_- = -theta_E
            Note: In two dimensions, this degenerate solution corresponds to an Einstein ring
            of radius theta_E rather than two isolated points.

        Args:
            beta: Signed scalar angular source offset (float or array).
            theta_e: Einstein radius (> 0).

        Returns:
            Tuple of (theta_plus, theta_minus) arrays.
        """
        if theta_e <= 0.0:
            raise ValueError(f"theta_e must be strictly positive, got {theta_e}.")

        beta_arr = np.asarray(beta, dtype=np.float64)
        discriminant = np.sqrt(beta_arr**2 + 4.0 * (theta_e**2))
        theta_plus = 0.5 * (beta_arr + discriminant)
        theta_minus = 0.5 * (beta_arr - discriminant)

        return theta_plus, theta_minus

    def image_positions_2d(
        self,
        beta_x: float,
        beta_y: float,
    ) -> dict:
        """Compute the 2D image positions for a point source at (beta_x, beta_y).

        Args:
            beta_x: Horizontal source plane position.
            beta_y: Vertical source plane position.

        Returns:
            Dictionary with keys:
                - 'is_ring' (bool): True if source is perfectly aligned with lens center.
                - 'einstein_ring_radius' (float): theta_E if is_ring is True.
                - 'center' (tuple): Lens center (x, y).
                - 'images' (list of tuples): [(x_plus, y_plus), (x_minus, y_minus)] if not perfectly aligned.
        """
        dx = beta_x - self._center[0]
        dy = beta_y - self._center[1]
        u = float(np.hypot(dx, dy))

        if np.isclose(u, 0.0, atol=1e-14):
            return {
                "is_ring": True,
                "einstein_ring_radius": self._theta_e,
                "center": self._center,
                "images": [],
                "description": (
                    "Perfect alignment: Source lies directly on the point caustic at the lens center. "
                    "Images form a continuous 1D Einstein ring with radius equal to the Einstein radius."
                ),
            }

        # Unit direction vector from lens center to source
        u_x = dx / u
        u_y = dy / u

        theta_plus_mag, theta_minus_mag = self.scalar_image_positions(u, self._theta_e)
        # Note: theta_minus_mag is negative, representing displacement in opposite direction
        img_plus = (
            float(self._center[0] + theta_plus_mag * u_x),
            float(self._center[1] + theta_plus_mag * u_y),
        )
        img_minus = (
            float(self._center[0] + theta_minus_mag * u_x),
            float(self._center[1] + theta_minus_mag * u_y),
        )

        return {
            "is_ring": False,
            "einstein_ring_radius": self._theta_e,
            "center": self._center,
            "images": [img_plus, img_minus],
            "description": "Two isolated point images (outer positive parity, inner negative parity).",
        }

    def image_magnifications_2d(
        self,
        beta_x: float,
        beta_y: float,
        allow_infinite: bool = False,
    ) -> dict:
        """Compute the signed, absolute, and total magnifications for a source at (beta_x, beta_y).

        Args:
            beta_x: Horizontal source plane position.
            beta_y: Vertical source plane position.
            allow_infinite: If True, returns infinite magnification when aligned.

        Returns:
            Dictionary containing 'mu_plus', 'mu_minus', 'abs_mu_plus', 'abs_mu_minus', 'mu_total'.
        """
        dx = beta_x - self._center[0]
        dy = beta_y - self._center[1]
        sep = float(np.hypot(dx, dy))
        u = sep / self._theta_e

        mu_plus, mu_minus = point_mass_magnification(u, allow_infinite=allow_infinite)
        abs_plus, abs_minus = point_mass_absolute_magnification(u, allow_infinite=allow_infinite)
        total_mu = point_mass_total_magnification(u, allow_infinite=allow_infinite)

        return {
            "u": u,
            "mu_plus": float(mu_plus),
            "mu_minus": float(mu_minus),
            "abs_mu_plus": float(abs_plus),
            "abs_mu_minus": float(abs_minus),
            "mu_total": float(total_mu),
        }
