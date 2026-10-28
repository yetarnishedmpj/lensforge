"""Singular Isothermal Sphere (SIS) gravitational lens model.

Mathematical Framework:
-----------------------
The Singular Isothermal Sphere represents an idealized, spherically symmetric galaxy
whose stellar velocity dispersion sigma_v is constant at all radii (flat rotation curve):

1. Three-Dimensional Density Profile:
   rho(r) = sigma_v^2 / (2 * pi * G * r^2)

2. Projected Surface Mass Density:
   Sigma(R) = sigma_v^2 / (2 * G * R)

3. Convergence (Dimensionless Surface Density):
   kappa(theta) = theta_E / (2 * |theta - theta_l|)

4. Einstein Radius:
   theta_E = 4 * pi * (sigma_v / c)^2 * (D_ls / D_s)

5. Reduced Deflection Angle:
   alpha(theta) = theta_E * (theta - theta_l) / |theta - theta_l|
   Note that the deflection magnitude |alpha| = theta_E is CONSTANT for all theta != theta_l.

6. Effective Lensing Potential:
   psi(theta) = theta_E * |theta - theta_l|
   Numerical and analytical gradient gives grad(psi) = alpha(theta).

7. Lens Equation & Image Solutions:
   beta = theta - alpha(theta)
   For collinear source offset u = |beta - theta_l|:
   - If u = 0: Perfect alignment produces a continuous Einstein ring of radius theta_E.
   - If 0 < u < theta_E: Two collinear images form on opposite sides of the lens:
     theta_+ = u + theta_E  (outer image, positive parity)
     theta_- = u - theta_E  (inner image, opposite side, negative parity)
     Image separation: theta_+ - theta_- = 2 * theta_E (independent of source position u!)
   - If u > theta_E: Only one image forms at theta_+ = u + theta_E.

8. Magnifications:
   mu_+ = 1 + theta_E / u = (u + theta_E) / u   (> 1, positive parity)
   mu_- = 1 - theta_E / u = - (theta_E - u) / u  (< 0, negative parity, for u < theta_E)
   Total magnification for u < theta_E: mu_tot = |mu_+| + |mu_-| = 2 * theta_E / u.
   Magnification invariant: mu_+ + mu_- = 2.0 across all valid 0 < u < theta_E.
"""

from typing import Optional, Tuple, Union
import numpy as np

from lensforge.constants import arcsec2rad, rad2arcsec
from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.distances import einstein_radius_sis
from lensforge.physics.lens_equation import lens_equation
from lensforge.physics.lens_potential import SingularityError
from lensforge.physics.models.base import BaseLensModel


class SingularIsothermalSphere(BaseLensModel):
    """Singular Isothermal Sphere (SIS) gravitational lens model.

    Attributes:
        theta_e: Einstein radius in radians (or consistent angular units).
        center: (center_x, center_y) coordinates of the lens center.
        velocity_dispersion: 1D velocity dispersion in m/s (if instantiated from physical parameters).
    """

    def __init__(
        self,
        theta_e: float,
        center: Tuple[float, float] = (0.0, 0.0),
        velocity_dispersion: Optional[float] = None,
        d_s: Optional[float] = None,
        d_ls: Optional[float] = None,
    ) -> None:
        """Initialize the SIS lens model.

        Args:
            theta_e: Einstein radius in radians (> 0).
            center: (center_x, center_y) lens center in radians.
            velocity_dispersion: Optional velocity dispersion in m/s.
            d_s: Optional observer-source angular-diameter distance in meters.
            d_ls: Optional lens-source angular-diameter distance in meters.

        Raises:
            ValueError: If theta_e <= 0.
        """
        if theta_e <= 0.0:
            raise ValueError(f"Einstein radius theta_e must be strictly positive, got {theta_e}.")

        self._theta_e = float(theta_e)
        self._center = (float(center[0]), float(center[1]))
        self._velocity_dispersion = float(velocity_dispersion) if velocity_dispersion is not None else None
        self._d_s = float(d_s) if d_s is not None else None
        self._d_ls = float(d_ls) if d_ls is not None else None

    @classmethod
    def from_velocity_dispersion(
        cls,
        velocity_dispersion: float,
        d_s: float,
        d_ls: float,
        center: Tuple[float, float] = (0.0, 0.0),
        strict_cosmology: bool = True,
    ) -> "SingularIsothermalSphere":
        """Instantiate an SIS lens from physical stellar velocity dispersion and cosmological distances.

        theta_E = 4 * pi * (sigma_v / c)^2 * (D_ls / D_s)

        Args:
            velocity_dispersion: 1D stellar velocity dispersion sigma_v in m/s (> 0).
            d_s: Observer-to-source angular-diameter distance in meters (> 0).
            d_ls: Lens-to-source angular-diameter distance in meters (> 0).
            center: (center_x, center_y) lens center in radians.
            strict_cosmology: If True, enforce cosmological ordering (d_ls < d_s).

        Returns:
            Configured SingularIsothermalSphere instance.
        """
        theta_e = einstein_radius_sis(
            velocity_dispersion=velocity_dispersion,
            d_s=d_s,
            d_ls=d_ls,
            strict_cosmology=strict_cosmology,
        )
        return cls(
            theta_e=theta_e,
            center=center,
            velocity_dispersion=velocity_dispersion,
            d_s=d_s,
            d_ls=d_ls,
        )

    @property
    def theta_e(self) -> float:
        """Einstein radius in radians."""
        return self._theta_e

    @property
    def theta_e_arcsec(self) -> float:
        """Einstein radius converted to arcseconds."""
        return float(rad2arcsec(self._theta_e))

    @property
    def center(self) -> Tuple[float, float]:
        """(center_x, center_y) coordinates of the lens center."""
        return self._center

    @property
    def velocity_dispersion(self) -> Optional[float]:
        """Stellar velocity dispersion in m/s (None if not provided)."""
        return self._velocity_dispersion

    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the 2D reduced deflection angle alpha(theta).

        alpha(theta) = theta_E * (theta - theta_l) / |theta - theta_l|

        Args:
            theta_x: Horizontal image-plane coordinate(s).
            theta_y: Vertical image-plane coordinate(s).
            mask_singular: If False, raises SingularityError at lens center.
                           If True, returns NaN at the singular point.

        Returns:
            Tuple of (alpha_x, alpha_y) deflection angles.

        Raises:
            SingularityError: If theta hits the lens center and mask_singular is False.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        r = np.hypot(dx, dy)

        is_singular = (r == 0.0)
        if np.any(is_singular):
            if not mask_singular:
                raise SingularityError(
                    f"SIS deflection field is singular at lens center {self._center}. "
                    "Set mask_singular=True to return NaN at singular coordinates."
                )

        with np.errstate(divide="ignore", invalid="ignore"):
            factor = self._theta_e / r
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
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the effective 2D gravitational lens potential psi(theta).

        psi(theta) = theta_E * |theta - theta_l|

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            mask_singular: Included for API compatibility with BaseLensModel.

        Returns:
            Lensing potential array.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        r = np.hypot(dx, dy)
        return self._theta_e * r

    def scalar_image_positions(
        self,
        beta: Union[float, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute the analytical collinear image positions for a scalar source offset beta.

        Given scalar source position beta along the radial line from the lens:
        - If |beta| < theta_E: Two images form:
            theta_+ = beta + theta_E  (outer image)
            theta_- = beta - theta_E  (inner image, negative parity)
        - If |beta| > theta_E: Only one image forms:
            theta_+ = beta + theta_E if beta > 0 (or beta - theta_E if beta < 0)
        - If beta = 0: Einstein ring of radius theta_E.

        Args:
            beta: Signed scalar angular source offset.

        Returns:
            Tuple of (theta_plus, theta_minus). theta_minus may be NaN if |beta| > theta_E.
        """
        b = np.asarray(beta, dtype=np.float64)
        abs_b = np.abs(b)

        # theta_+ always exists on the same side as beta: sign(beta) * (|beta| + theta_E)
        sgn = np.where(b >= 0.0, 1.0, -1.0)
        theta_plus = sgn * (abs_b + self._theta_e)

        # theta_- only exists if |beta| < theta_E
        # for scalar beta > 0: theta_- = beta - theta_E (< 0, opposite side)
        with np.errstate(invalid="ignore"):
            theta_minus = np.where(
                abs_b < self._theta_e,
                sgn * (abs_b - self._theta_e),
                np.nan,
            )

        return theta_plus, theta_minus

    def image_positions_2d(
        self,
        beta_x: float,
        beta_y: float,
    ) -> dict:
        """Compute 2D image positions for a point source at (beta_x, beta_y).

        Args:
            beta_x: Horizontal source plane position.
            beta_y: Vertical source plane position.

        Returns:
            Dictionary describing image positions and geometry:
                - 'is_ring' (bool): True if perfectly aligned (Einstein ring).
                - 'einstein_ring_radius' (float): theta_E.
                - 'images' (list of tuples): [(x+, y+), (x-, y-)] or single image [(x+, y+)].
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
                "description": "Perfect alignment: Source on central caustic forms a continuous Einstein ring.",
            }

        u_x = dx / u
        u_y = dy / u

        # Outer image always forms at distance u + theta_E from center in direction of source
        r_plus = u + self._theta_e
        img_plus = (
            float(self._center[0] + r_plus * u_x),
            float(self._center[1] + r_plus * u_y),
        )

        if u < self._theta_e:
            # Inner image forms at distance theta_E - u in opposite direction
            r_minus = self._theta_e - u
            img_minus = (
                float(self._center[0] - r_minus * u_x),
                float(self._center[1] - r_minus * u_y),
            )
            images = [img_plus, img_minus]
            desc = "Two collinear images (outer image, inner image with reversed parity)."
        else:
            images = [img_plus]
            desc = "Single image (source lies outside radial caustic |beta| > theta_E)."

        return {
            "is_ring": False,
            "einstein_ring_radius": self._theta_e,
            "center": self._center,
            "images": images,
            "description": desc,
        }

    def image_magnifications_2d(
        self,
        beta_x: float,
        beta_y: float,
        allow_infinite: bool = False,
    ) -> dict:
        """Compute signed, absolute, and total magnifications for a source at (beta_x, beta_y).

        mu_+ = 1 + theta_E / u
        mu_- = 1 - theta_E / u  (if u < theta_E)

        Args:
            beta_x: Horizontal source plane position.
            beta_y: Vertical source plane position.
            allow_infinite: If True, returns infinite magnification when aligned.

        Returns:
            Dictionary with 'u', 'mu_plus', 'mu_minus', 'abs_mu_plus', 'abs_mu_minus', 'mu_total'.

        Raises:
            ZeroDivisionError: If u == 0 and allow_infinite is False.
        """
        dx = beta_x - self._center[0]
        dy = beta_y - self._center[1]
        u = float(np.hypot(dx, dy))

        if np.isclose(u, 0.0, atol=1e-14):
            if not allow_infinite:
                raise ZeroDivisionError(
                    "Source lies at caustic center u = 0. Magnification diverges to infinity. "
                    "Set allow_infinite=True to return np.inf."
                )
            return {
                "u": 0.0,
                "mu_plus": float(np.inf),
                "mu_minus": float(-np.inf),
                "abs_mu_plus": float(np.inf),
                "abs_mu_minus": float(np.inf),
                "mu_total": float(np.inf),
            }

        ratio = self._theta_e / u
        mu_plus = 1.0 + ratio
        abs_plus = mu_plus

        if u < self._theta_e:
            mu_minus = 1.0 - ratio  # negative because ratio > 1
            abs_minus = abs(mu_minus)
            mu_total = abs_plus + abs_minus  # = 2 * theta_E / u
        else:
            mu_minus = None
            abs_minus = 0.0
            mu_total = abs_plus

        return {
            "u": u,
            "mu_plus": float(mu_plus),
            "mu_minus": float(mu_minus) if mu_minus is not None else None,
            "abs_mu_plus": float(abs_plus),
            "abs_mu_minus": float(abs_minus),
            "mu_total": float(mu_total),
        }

