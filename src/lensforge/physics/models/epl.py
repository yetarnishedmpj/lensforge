"""Elliptical Power-Law (EPL / PEMD) gravitational lens model.

Mathematical Framework:
-----------------------
The Elliptical Power-Law model generalizes isothermal profiles (SIE/SIS) to galaxies
with arbitrary 3D radial density slopes gamma_pl in (1, 3):

    rho(r) = rho_0 * (r_0 / r)^gamma_pl

Astrophysical Significance:
---------------------------
- gamma_pl = 2.0: Isothermal profile (flat rotation curves, identical to SIE/SIS).
- gamma_pl < 2.0: Shallow density cusp / cored profile (lower central dark matter density).
- gamma_pl > 2.0: Steep density cusp (contracted dark matter halo or compact stellar core).

Using the elliptical potential formulation (Kassiola & Kovner 1993, Chae et al. 1998, Keeton 2001):

Let (theta_x, theta_y) be celestial coordinates and (x_l, y_l) the lens center.
Rotate to the lens major-axis frame by position angle phi:
    x_1 =  cos(phi) * (theta_x - x_l) + sin(phi) * (theta_y - y_l)
    x_2 = -sin(phi) * (theta_x - x_l) + cos(phi) * (theta_y - y_l)

Define the elliptical radius:
    r_e = sqrt( q^2 * x_1^2 + x_2^2 + s^2 )
where:
    q is the minor-to-major axis ratio (0 < q <= 1).
    s is an optional core radius to regularize the central singularity (default 0.0).

1. Lensing Potential psi(x_1, x_2):
    psi = [ theta_E^2 / (3 - gamma_pl) ] * ( r_e / theta_E )^(3 - gamma_pl)

2. Deflection Angle in Principal Frame:
    alpha_1 = d(psi)/d(x_1) = q^2 * x_1 * theta_E^(gamma_pl - 1) * r_e^(1 - gamma_pl)
    alpha_2 = d(psi)/d(x_2) =       x_2 * theta_E^(gamma_pl - 1) * r_e^(1 - gamma_pl)

3. Rotate Deflection back to Celestial Coordinates:
    alpha_x =  cos(phi) * alpha_1 - sin(phi) * alpha_2
    alpha_y =  sin(phi) * alpha_1 + cos(phi) * alpha_2

This model is completely analytical, SI-consistent, and has machine-precision derivatives.
"""

from typing import Tuple, Union
import numpy as np

from lensforge.constants import deg2rad
from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.models.base import BaseLensModel


class EllipticalPowerLaw(BaseLensModel):
    """Elliptical Power-Law (EPL / PEMD) gravitational lens model.

    Attributes:
        theta_e: Characteristic Einstein radius in angular units (> 0).
        gamma: 3D mass density slope gamma in (1.0, 3.0), where 2.0 is isothermal.
        axis_ratio: Minor-to-major axis ratio q in (0, 1].
        pos_angle: Major axis position angle phi in radians (counterclockwise from +x).
        core_radius: Core softening radius s (default 1e-6 to avoid singularity at origin).
        center: (center_x, center_y) coordinates of the lens center.
    """

    def __init__(
        self,
        theta_e: float,
        gamma: float = 2.0,
        axis_ratio: float = 1.0,
        pos_angle: float = 0.0,
        core_radius: float = 1e-6,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize the Elliptical Power-Law lens model.

        Args:
            theta_e: Einstein radius (> 0).
            gamma: 3D power-law slope, must satisfy 1.0 < gamma < 3.0 (default 2.0).
            axis_ratio: Minor-to-major axis ratio q in (0, 1].
            pos_angle: Major axis position angle in radians.
            core_radius: Finite core radius s >= 0.
            center: (center_x, center_y) lens center.
        """
        if theta_e <= 0.0:
            raise ValueError(f"Einstein radius theta_e must be strictly positive, got {theta_e}.")
        if not (1.0 < gamma < 3.0):
            raise ValueError(f"Power-law slope gamma must be in (1.0, 3.0), got {gamma}.")
        if not (0.0 < axis_ratio <= 1.0):
            raise ValueError(f"Axis ratio q must be in (0, 1], got {axis_ratio}.")
        if core_radius < 0.0:
            raise ValueError(f"Core radius must be non-negative, got {core_radius}.")

        self._theta_e = float(theta_e)
        self._gamma = float(gamma)
        self._axis_ratio = float(axis_ratio)
        self._pos_angle = float(pos_angle)
        self._core_radius = float(core_radius)
        self._center = (float(center[0]), float(center[1]))

        # Precompute constants
        self._cos_phi = float(np.cos(self._pos_angle))
        self._sin_phi = float(np.sin(self._pos_angle))
        self._q2 = float(self._axis_ratio**2)
        self._s2 = float(self._core_radius**2)
        self._eta = 3.0 - self._gamma  # Exponent for potential: (r_e/theta_e)^eta
        self._norm_pot = (self._theta_e**2) / self._eta
        self._norm_def = self._theta_e ** (self._gamma - 1.0)

    @classmethod
    def from_degrees(
        cls,
        theta_e: float,
        gamma: float = 2.0,
        axis_ratio: float = 1.0,
        pos_angle_deg: float = 0.0,
        core_radius: float = 1e-6,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> "EllipticalPowerLaw":
        """Instantiate an EllipticalPowerLaw with position angle in degrees."""
        return cls(
            theta_e=theta_e,
            gamma=gamma,
            axis_ratio=axis_ratio,
            pos_angle=deg2rad(pos_angle_deg),
            core_radius=core_radius,
            center=center,
        )

    @property
    def theta_e(self) -> float:
        """Characteristic Einstein radius."""
        return self._theta_e

    @property
    def gamma(self) -> float:
        """3D mass density slope."""
        return self._gamma

    @property
    def axis_ratio(self) -> float:
        """Minor-to-major axis ratio q."""
        return self._axis_ratio

    @property
    def pos_angle(self) -> float:
        """Position angle phi in radians."""
        return self._pos_angle

    @property
    def center(self) -> Tuple[float, float]:
        """Lens centroid (center_x, center_y)."""
        return self._center

    def _rotate_to_principal(
        self,
        theta_x: np.ndarray,
        theta_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate celestial coordinates to the lens principal-axis frame."""
        dx = theta_x - self._center[0]
        dy = theta_y - self._center[1]
        x1 = self._cos_phi * dx + self._sin_phi * dy
        x2 = -self._sin_phi * dx + self._cos_phi * dy
        return x1, x2

    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the 2D reduced deflection angle alpha(theta)."""
        tx, ty = validate_coordinates(theta_x, theta_y)
        x1, x2 = self._rotate_to_principal(tx, ty)

        r_e2 = self._q2 * (x1**2) + (x2**2) + self._s2
        r_e = np.sqrt(np.maximum(1e-12, r_e2))

        # In principal frame:
        # alpha_1 = q^2 * x1 * theta_E^(gamma - 1) * r_e^(1 - gamma)
        # alpha_2 =       x2 * theta_E^(gamma - 1) * r_e^(1 - gamma)
        radial_factor = self._norm_def * (r_e ** (1.0 - self._gamma))
        a1 = self._q2 * x1 * radial_factor
        a2 = x2 * radial_factor

        # Rotate back to celestial frame:
        # alpha_x =  cos(phi) * a1 - sin(phi) * a2
        # alpha_y =  sin(phi) * a1 + cos(phi) * a2
        ax = self._cos_phi * a1 - self._sin_phi * a2
        ay = self._sin_phi * a1 + self._cos_phi * a2
        return ax, ay

    def potential(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the 2D effective gravitational potential psi(theta)."""
        tx, ty = validate_coordinates(theta_x, theta_y)
        x1, x2 = self._rotate_to_principal(tx, ty)

        r_e2 = self._q2 * (x1**2) + (x2**2) + self._s2
        r_e = np.sqrt(np.maximum(1e-12, r_e2))

        # psi = [ theta_E^2 / (3 - gamma) ] * ( r_e / theta_E )^(3 - gamma)
        return self._norm_pot * ((r_e / self._theta_e) ** self._eta)
