"""Singular Isothermal Ellipsoid (SIE) gravitational lens model.

Mathematical Framework:
-----------------------
The Singular Isothermal Ellipsoid (SIE) is the premier parametric lens model used in
strong gravitational lensing of elliptical galaxies, providing flat rotation curves
while capturing the elliptical shape (flattening) of real galaxy mass distributions.

Formulated by Kormann, Schneider & Bartelmann (1994) [KSB94] and Keeton (2001):

1. Coordinates & Orientation:
   Let (theta_x, theta_y) be sky coordinates and (x_l, y_l) the lens centroid.
   Let q = b/a (0 < q <= 1) be the minor-to-major axis ratio.
   Let phi be the position angle of the major axis measured counterclockwise from the +x axis.
   
   Rotation to lens principal-axis coordinates:
       x_1 =  cos(phi) * (theta_x - x_l) + sin(phi) * (theta_y - y_l)
       x_2 = -sin(phi) * (theta_x - x_l) + cos(phi) * (theta_y - y_l)

2. Elliptical Radius:
   r = sqrt( q^2 * x_1^2 + x_2^2 )
   s = sqrt( 1 - q^2 )   (eccentricity parameter)

3. Convergence (Dimensionless Surface Mass Density):
   kappa(x_1, x_2) = (theta_E * sqrt(q)) / (2 * r)

4. Deflection Angle in Principal-Axis Frame:
   For q < 1:
       alpha_1(x_1, x_2) = [theta_E * sqrt(q) / s] * arctan( s * x_1 / r )
       alpha_2(x_1, x_2) = [theta_E * sqrt(q) / s] * artanh( s * x_2 / r )
   
   Smooth Spherical Limit (q -> 1):
       alpha_1 -> theta_E * x_1 / sqrt(x_1^2 + x_2^2)
       alpha_2 -> theta_E * x_2 / sqrt(x_1^2 + x_2^2)   (identically SIS)

5. Rotation to Celestial Coordinates:
   alpha_x =  cos(phi) * alpha_1 - sin(phi) * alpha_2
   alpha_y =  sin(phi) * alpha_1 + cos(phi) * alpha_2

6. Effective Lensing Potential & Euler's Homogeneous Function Theorem:
   Because the isothermal potential is scale-free (homogeneous of degree 1):
       psi(x_1, x_2) = x_1 * alpha_1 + x_2 * alpha_2 = Delta_x * alpha_x + Delta_y * alpha_y
   By Euler's theorem:
       grad(psi) = (alpha_x, alpha_y) exactly across all smooth regions!
"""

from typing import Optional, Tuple, Union
import numpy as np

from lensforge.constants import arcsec2rad, deg2rad, rad2arcsec, rad2deg
from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.lens_equation import lens_equation
from lensforge.physics.lens_potential import SingularityError
from lensforge.physics.models.base import BaseLensModel


class SingularIsothermalEllipsoid(BaseLensModel):
    """Singular Isothermal Ellipsoid (SIE) gravitational lens model.

    Attributes:
        theta_e: Characteristic Einstein radius in radians (or consistent angular units).
        axis_ratio: Minor-to-major axis ratio q = b/a (0 < q <= 1).
        pos_angle: Position angle phi in radians of the major axis.
        center: (center_x, center_y) coordinates of the lens center.
    """

    def __init__(
        self,
        theta_e: float,
        axis_ratio: float,
        pos_angle: float = 0.0,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize the SIE lens model.

        Args:
            theta_e: Einstein radius in radians (> 0).
            axis_ratio: Axis ratio q = b/a, must satisfy 0 < q <= 1.
            pos_angle: Position angle phi of major axis in radians (counterclockwise from +x).
            center: (center_x, center_y) lens center coordinates.

        Raises:
            ValueError: If theta_e <= 0 or not (0 < axis_ratio <= 1).
        """
        if theta_e <= 0.0:
            raise ValueError(f"Einstein radius theta_e must be strictly positive, got {theta_e}.")
        if not (0.0 < axis_ratio <= 1.0):
            raise ValueError(f"Axis ratio q must be in (0, 1], got {axis_ratio}.")

        self._theta_e = float(theta_e)
        self._axis_ratio = float(axis_ratio)
        self._pos_angle = float(pos_angle)
        self._center = (float(center[0]), float(center[1]))

        # Precompute trigonometry
        self._cos_phi = float(np.cos(self._pos_angle))
        self._sin_phi = float(np.sin(self._pos_angle))
        self._s = float(np.sqrt(max(0.0, 1.0 - self._axis_ratio**2)))

    @classmethod
    def from_degrees(
        cls,
        theta_e: float,
        axis_ratio: float,
        pos_angle_deg: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> "SingularIsothermalEllipsoid":
        """Instantiate an SIE with position angle specified in degrees.

        Args:
            theta_e: Einstein radius in radians (or arcsec if consistent).
            axis_ratio: Axis ratio q in (0, 1].
            pos_angle_deg: Major axis position angle in degrees.
            center: Lens center (x, y).

        Returns:
            SingularIsothermalEllipsoid instance.
        """
        return cls(
            theta_e=theta_e,
            axis_ratio=axis_ratio,
            pos_angle=float(deg2rad(pos_angle_deg)),
            center=center,
        )

    @property
    def theta_e(self) -> float:
        """Einstein radius."""
        return self._theta_e

    @property
    def theta_e_arcsec(self) -> float:
        """Einstein radius converted to arcseconds."""
        return float(rad2arcsec(self._theta_e))

    @property
    def axis_ratio(self) -> float:
        """Minor-to-major axis ratio q."""
        return self._axis_ratio

    @property
    def pos_angle(self) -> float:
        """Position angle phi in radians."""
        return self._pos_angle

    @property
    def pos_angle_deg(self) -> float:
        """Position angle phi converted to degrees."""
        return float(rad2deg(self._pos_angle))

    @property
    def center(self) -> Tuple[float, float]:
        """(center_x, center_y) coordinates of the lens center."""
        return self._center

    def _rotate_to_principal_axes(
        self,
        dx: np.ndarray,
        dy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate coordinates relative to lens center into principal axes frame."""
        x1 = self._cos_phi * dx + self._sin_phi * dy
        x2 = -self._sin_phi * dx + self._cos_phi * dy
        return x1, x2

    def _rotate_from_principal_axes(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate vector from principal axes frame back to celestial frame."""
        ax = self._cos_phi * a1 - self._sin_phi * a2
        ay = self._sin_phi * a1 + self._cos_phi * a2
        return ax, ay

    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the 2D reduced deflection angle alpha(theta) via KSB94 formulas.

        Args:
            theta_x: Horizontal image-plane coordinate(s).
            theta_y: Vertical image-plane coordinate(s).
            mask_singular: If False, raise SingularityError at lens center.
                           If True, return NaN at the singular point.

        Returns:
            Tuple of (alpha_x, alpha_y) deflection angles.

        Raises:
            SingularityError: If theta hits the lens center and mask_singular is False.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]

        # Rotate to principal axes
        x1, x2 = self._rotate_to_principal_axes(dx, dy)
        r_sph = np.hypot(x1, x2)

        is_singular = (r_sph == 0.0)
        if np.any(is_singular):
            if not mask_singular:
                raise SingularityError(
                    f"SIE deflection field is singular at lens center {self._center}. "
                    "Set mask_singular=True to return NaN at singular coordinates."
                )

        q = self._axis_ratio
        s = self._s

        # Nearly spherical branch (q -> 1 limit avoids numerical 0/0)
        if s < 1e-5:
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = self._theta_e / r_sph
                a1 = factor * x1
                a2 = factor * x2
        else:
            # Elliptical radius r = sqrt(q^2 * x1^2 + x2^2)
            r = np.sqrt(q**2 * x1**2 + x2**2)
            with np.errstate(divide="ignore", invalid="ignore"):
                prefactor = (self._theta_e * np.sqrt(q)) / s
                arg_x1 = (s * x1) / r
                arg_x2 = (s * x2) / r
                # Clamping arg_x2 within (-1 + eps, 1 - eps) prevents inf in arctanh
                # Note: analytically, |s * x2 / r| <= s < 1 whenever q > 0
                a1 = prefactor * np.arctan(arg_x1)
                a2 = prefactor * np.arctanh(np.clip(arg_x2, -0.999999999999, 0.999999999999))

        if np.any(is_singular):
            a1 = np.where(is_singular, np.nan, a1)
            a2 = np.where(is_singular, np.nan, a2)

        # Rotate back to celestial frame
        ax, ay = self._rotate_from_principal_axes(a1, a2)
        return ax, ay

    def potential(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the effective 2D gravitational lens potential psi(theta).

        By Euler's theorem for homogeneous functions of degree 1:
            psi(theta) = Delta_x * alpha_x + Delta_y * alpha_y

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            mask_singular: Singularity masking policy passed to deflection.

        Returns:
            Lensing potential array.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        ax, ay = self.deflection(theta_x, theta_y, mask_singular=mask_singular)
        return dx * ax + dy * ay

    def convergence_analytic(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the analytical surface mass density / convergence kappa(theta).

        kappa = (theta_E * sqrt(q)) / (2 * sqrt(q^2 * x_1^2 + x_2^2))

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            mask_singular: Singularity masking policy.

        Returns:
            Analytical convergence array.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        x1, x2 = self._rotate_to_principal_axes(dx, dy)
        r = np.sqrt(self._axis_ratio**2 * x1**2 + x2**2)

        is_singular = (r == 0.0)
        if np.any(is_singular):
            if not mask_singular:
                raise SingularityError("SIE convergence is singular at the center.")

        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = (self._theta_e * np.sqrt(self._axis_ratio)) / (2.0 * r)
            if np.any(is_singular):
                kappa = np.where(is_singular, np.nan, kappa)

        return kappa

