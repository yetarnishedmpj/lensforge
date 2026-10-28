"""External gravitational shear model.

Mathematical Framework:
-----------------------
External shear represents the tidal gravitational perturbations induced on a strong lens
by neighboring galaxies, group halos, or large-scale structure (cluster environment):

1. Parameterization:
   Shear components (gamma_1, gamma_2), or magnitude gamma and orientation phi_gamma:
       gamma_1 = gamma * cos(2 * phi_gamma)
       gamma_2 = gamma * sin(2 * phi_gamma)
       gamma = sqrt(gamma_1^2 + gamma_2^2)

2. Effective Lensing Potential:
   psi(theta) = 1/2 * gamma_1 * (Delta_x^2 - Delta_y^2) + gamma_2 * Delta_x * Delta_y

3. Reduced Deflection Angle:
   alpha_x = d(psi)/d(theta_x) = gamma_1 * Delta_x + gamma_2 * Delta_y
   alpha_y = d(psi)/d(theta_y) = gamma_2 * Delta_x - gamma_1 * Delta_y

4. Massless Tidal Field (Zero Convergence):
   kappa = 1/2 * (d(alpha_x)/dx + d(alpha_y)/dy) = 1/2 * (gamma_1 - gamma_1) = 0.0

5. Constant Magnification:
   det(A) = (1 - gamma_1) * (1 + gamma_1) - gamma_2^2 = 1 - (gamma_1^2 + gamma_2^2) = 1 - gamma^2
   mu = 1 / (1 - gamma^2)
"""

from typing import Optional, Tuple, Union
import numpy as np

from lensforge.constants import deg2rad, rad2deg
from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.models.base import BaseLensModel


class ExternalShear(BaseLensModel):
    """External gravitational shear lens model.

    Attributes:
        gamma_1: Shear component along coordinate axes.
        gamma_2: Shear component along 45-degree axes.
        center: (center_x, center_y) expansion center coordinates.
    """

    def __init__(
        self,
        gamma_1: float,
        gamma_2: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize the external shear model from Cartesian shear components.

        Args:
            gamma_1: First shear component.
            gamma_2: Second shear component.
            center: (center_x, center_y) expansion origin.
        """
        self._gamma_1 = float(gamma_1)
        self._gamma_2 = float(gamma_2)
        self._center = (float(center[0]), float(center[1]))

    @classmethod
    def from_polar(
        cls,
        gamma: float,
        phi_gamma: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> "ExternalShear":
        """Instantiate external shear from shear magnitude and position angle in radians.

        gamma_1 = gamma * cos(2 * phi_gamma)
        gamma_2 = gamma * sin(2 * phi_gamma)

        Args:
            gamma: Shear magnitude (|gamma| >= 0).
            phi_gamma: Shear orientation angle in radians.
            center: Lens center.

        Returns:
            ExternalShear instance.
        """
        g = float(gamma)
        if g < 0.0:
            raise ValueError(f"Shear magnitude gamma must be non-negative, got {gamma}.")
        g1 = g * np.cos(2.0 * phi_gamma)
        g2 = g * np.sin(2.0 * phi_gamma)
        return cls(gamma_1=g1, gamma_2=g2, center=center)

    @classmethod
    def from_polar_degrees(
        cls,
        gamma: float,
        phi_gamma_deg: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> "ExternalShear":
        """Instantiate external shear from magnitude and angle in degrees."""
        return cls.from_polar(
            gamma=gamma,
            phi_gamma=float(deg2rad(phi_gamma_deg)),
            center=center,
        )

    @property
    def gamma_1(self) -> float:
        """First shear component."""
        return self._gamma_1

    @property
    def gamma_2(self) -> float:
        """Second shear component."""
        return self._gamma_2

    @property
    def gamma(self) -> float:
        """Total shear magnitude gamma = sqrt(gamma_1^2 + gamma_2^2)."""
        return float(np.hypot(self._gamma_1, self._gamma_2))

    @property
    def phi_gamma(self) -> float:
        """Shear position angle in radians: 0.5 * atan2(gamma_2, gamma_1)."""
        return float(0.5 * np.arctan2(self._gamma_2, self._gamma_1))

    @property
    def phi_gamma_deg(self) -> float:
        """Shear position angle in degrees."""
        return float(rad2deg(self.phi_gamma))

    @property
    def center(self) -> Tuple[float, float]:
        """Expansion origin."""
        return self._center

    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the shear deflection vector alpha(theta).

        alpha_x = gamma_1 * Delta_x + gamma_2 * Delta_y
        alpha_y = gamma_2 * Delta_x - gamma_1 * Delta_y

        Args:
            theta_x: Horizontal image-plane coordinate(s).
            theta_y: Vertical image-plane coordinate(s).
            mask_singular: Included for BaseLensModel compliance (shear is non-singular everywhere).

        Returns:
            Tuple of (alpha_x, alpha_y) arrays.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]

        alpha_x = self._gamma_1 * dx + self._gamma_2 * dy
        alpha_y = self._gamma_2 * dx - self._gamma_1 * dy
        return alpha_x, alpha_y

    def potential(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the shear lensing potential psi(theta).

        psi = 0.5 * gamma_1 * (Delta_x^2 - Delta_y^2) + gamma_2 * Delta_x * Delta_y

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            mask_singular: Included for BaseLensModel compliance.

        Returns:
            Potential array.
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]

        return 0.5 * self._gamma_1 * (dx**2 - dy**2) + self._gamma_2 * (dx * dy)

