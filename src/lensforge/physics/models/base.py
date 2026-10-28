"""Base class and interface for parametric gravitational lens models.

This module defines the abstract base class `BaseLensModel` from which all analytical
and numerical lens models inherit.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np

from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.lens_equation import lens_equation


class BaseLensModel(ABC):
    """Abstract base class for all gravitational lens models in LensForge.

    Subclasses must implement:
    - deflection(theta_x, theta_y, mask_singular=False)
    - potential(theta_x, theta_y, mask_singular=False)

    Provides built-in default implementations for:
    - ray_trace(theta_x, theta_y, mask_singular=False)
    - jacobian(theta_x, theta_y, eps=1e-6)
    - magnification(theta_x, theta_y, eps=1e-6)
    - convergence(theta_x, theta_y, eps=1e-6)
    - shear(theta_x, theta_y, eps=1e-6)
    """

    @abstractmethod
    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the 2D reduced deflection angle alpha(theta).

        Args:
            theta_x: Image-plane horizontal angular coordinate(s).
            theta_y: Image-plane vertical angular coordinate(s).
            mask_singular: If True, replace singular points with NaN instead of raising.

        Returns:
            Tuple of (alpha_x, alpha_y) deflection angles.
        """

    @abstractmethod
    def potential(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the 2D effective gravitational lens potential psi(theta).

        Args:
            theta_x: Image-plane horizontal angular coordinate(s).
            theta_y: Image-plane vertical angular coordinate(s).
            mask_singular: If True, replace singular points with NaN instead of raising.

        Returns:
            Lensing potential array.
        """

    def ray_trace(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map image plane coordinates theta back to source plane coordinates beta via lens equation.

        beta = theta - alpha(theta)

        Args:
            theta_x: Image-plane horizontal coordinate(s).
            theta_y: Image-plane vertical coordinate(s).
            mask_singular: Singularity masking policy passed to deflection.

        Returns:
            Tuple of (beta_x, beta_y) source coordinates.
        """
        alpha_x, alpha_y = self.deflection(theta_x, theta_y, mask_singular=mask_singular)
        return lens_equation(theta_x, theta_y, alpha_x, alpha_y)

    def jacobian(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the 2x2 lensing Jacobian matrix A = I - d(alpha)/d(theta) via central finite differences.

        A = [[A_xx, A_xy],
             [A_yx, A_yy]]
        where:
            A_xx = 1 - d(alpha_x)/d(theta_x)
            A_xy =   - d(alpha_x)/d(theta_y)
            A_yx =   - d(alpha_y)/d(theta_x)
            A_yy = 1 - d(alpha_y)/d(theta_y)

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            eps: Finite-difference step size in radians (or arcsec, matching coordinates).

        Returns:
            Tuple of (A_xx, A_xy, A_yx, A_yy).
        """
        tx, ty = validate_coordinates(theta_x, theta_y)

        ax_xp, ay_xp = self.deflection(tx + eps, ty, mask_singular=True)
        ax_xm, ay_xm = self.deflection(tx - eps, ty, mask_singular=True)
        ax_yp, ay_yp = self.deflection(tx, ty + eps, mask_singular=True)
        ax_ym, ay_ym = self.deflection(tx, ty - eps, mask_singular=True)

        inv_2eps = 1.0 / (2.0 * eps)
        dax_dx = (ax_xp - ax_xm) * inv_2eps
        day_dx = (ay_xp - ay_xm) * inv_2eps
        dax_dy = (ax_yp - ax_ym) * inv_2eps
        day_dy = (ay_yp - ay_ym) * inv_2eps

        a_xx = 1.0 - dax_dx
        a_xy = -dax_dy
        a_yx = -day_dx
        a_yy = 1.0 - day_dy

        return a_xx, a_xy, a_yx, a_yy

    def magnification(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute the signed magnification mu(theta) = 1 / det(A).

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            eps: Finite-difference step size.

        Returns:
            Signed magnification array.
        """
        a_xx, a_xy, a_yx, a_yy = self.jacobian(theta_x, theta_y, eps=eps)
        det_a = a_xx * a_yy - a_xy * a_yx
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 1.0 / det_a
        return mu

    def convergence(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute the dimensionless surface mass density / convergence kappa(theta).

        kappa = 1 - 1/2 * (A_xx + A_yy) = 1/2 * (d(alpha_x)/dx + d(alpha_y)/dy)

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            eps: Finite-difference step size.

        Returns:
            Convergence array.
        """
        a_xx, _, _, a_yy = self.jacobian(theta_x, theta_y, eps=eps)
        return 1.0 - 0.5 * (a_xx + a_yy)

    def shear(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gravitational shear components (gamma_1, gamma_2).

        gamma_1 = 1/2 * (A_yy - A_xx)
        gamma_2 = - 1/2 * (A_xy + A_yx)

        Args:
            theta_x: Horizontal coordinate(s).
            theta_y: Vertical coordinate(s).
            eps: Finite-difference step size.

        Returns:
            Tuple of (gamma_1, gamma_2).
        """
        a_xx, a_xy, a_yx, a_yy = self.jacobian(theta_x, theta_y, eps=eps)
        gamma_1 = 0.5 * (a_yy - a_xx)
        gamma_2 = -0.5 * (a_xy + a_yx)
        return gamma_1, gamma_2

