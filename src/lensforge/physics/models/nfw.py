"""Navarro-Frenk-White (NFW) dark matter halo gravitational lens model.

Mathematical Framework:
-----------------------
The NFW profile (Navarro, Frenk & White 1996, 1997) is the universal density profile
of cold dark matter (CDM) halos produced across cosmological N-body simulations:

    rho(r) = rho_s / [ (r / r_s) * (1 + r / r_s)^2 ]

where:
    r_s is the characteristic scale radius where d ln(rho)/d ln(r) = -2.
    rho_s is the characteristic central density.

In gravitational lensing, the analytical projected surface mass density (convergence),
deflection angle, and lensing potential were derived in closed form by Bartelmann (1996)
and Wright & Brainerd (2000).

Let theta_s = r_s / D_l be the angular scale radius on the sky, and define the
dimensionless angular radius:
    x = theta / theta_s

1. Convergence kappa(x) = Sigma(theta) / Sigma_cr:
    kappa(x) = 2 * kappa_s * f(x) / (x^2 - 1)
    where kappa_s = (rho_s * r_s) / Sigma_cr is the characteristic dimensionless convergence,
    and:
        f(x) = 1 - (2 / sqrt(1 - x^2)) * artanh(sqrt((1 - x) / (1 + x)))   for x < 1
        f(1) = 0  (smooth limit kappa(1) = 2/3 * kappa_s)
        f(x) = 1 - (2 / sqrt(x^2 - 1)) * arctan(sqrt((x - 1) / (1 + x)))   for x > 1

2. Deflection Angle alpha(x):
    alpha(x) = 4 * kappa_s * theta_s * (g(x) / x) * (theta / theta_abs)
    where:
        g(x) = ln(x / 2) + (2 / sqrt(1 - x^2)) * artanh(sqrt((1 - x) / (1 + x)))   for x < 1
        g(1) = 1 + ln(1 / 2)
        g(x) = ln(x / 2) + (2 / sqrt(x^2 - 1)) * arctan(sqrt((x - 1) / (1 + x)))   for x > 1

3. Lensing Potential psi(x):
    psi(x) = 2 * kappa_s * theta_s^2 * h(x)
    where:
        h(x) = 0.5 * ln^2(x / 2) - 2 * artanh^2(sqrt((1 - x) / (1 + x)))   for x < 1
        h(1) = 0.5 * ln^2(0.5)
        h(x) = 0.5 * ln^2(x / 2) + 2 * arctan^2(sqrt((x - 1) / (1 + x)))   for x > 1

Notice that unlike singular isothermal profiles, the NFW deflection alpha -> 0 smoothly
at the center, because the 3D density cusp is only ~ 1/r, meaning enclosed mass
M(<= R) -> 0 faster than R.
"""

from typing import Tuple, Union
import numpy as np

from lensforge.physics.coordinates import validate_coordinates
from lensforge.physics.lens_equation import lens_equation
from lensforge.physics.models.base import BaseLensModel


def _nfw_f(x: np.ndarray) -> np.ndarray:
    """Helper computing the NFW convergence radial kernel f(x) / (x^2 - 1)."""
    res = np.zeros_like(x, dtype=np.float64)
    # x < 1
    m_inner = x < 1.0 - 1e-6
    if np.any(m_inner):
        xi = x[m_inner]
        sq = np.sqrt(np.maximum(0.0, 1.0 - xi**2))
        res[m_inner] = (1.0 - (2.0 / sq) * np.arctanh(np.sqrt((1.0 - xi) / (1.0 + xi)))) / (xi**2 - 1.0)

    # x > 1
    m_outer = x > 1.0 + 1e-6
    if np.any(m_outer):
        xo = x[m_outer]
        sq = np.sqrt(np.maximum(0.0, xo**2 - 1.0))
        res[m_outer] = (1.0 - (2.0 / sq) * np.arctan(np.sqrt((xo - 1.0) / (1.0 + xo)))) / (xo**2 - 1.0)

    # x near 1 (smooth Taylor limit: kappa(1) = 2/3 * kappa_s, so kernel = 1/3)
    m_mid = (~m_inner) & (~m_outer)
    if np.any(m_mid):
        res[m_mid] = 1.0 / 3.0

    return res


def _nfw_g(x: np.ndarray) -> np.ndarray:
    """Helper computing the NFW deflection radial kernel g(x)."""
    res = np.zeros_like(x, dtype=np.float64)
    # x < 1
    m_inner = (x < 1.0 - 1e-6) & (x > 1e-7)
    if np.any(m_inner):
        xi = x[m_inner]
        sq = np.sqrt(np.maximum(0.0, 1.0 - xi**2))
        res[m_inner] = np.log(xi / 2.0) + (2.0 / sq) * np.arctanh(np.sqrt((1.0 - xi) / (1.0 + xi)))

    # x > 1
    m_outer = x > 1.0 + 1e-6
    if np.any(m_outer):
        xo = x[m_outer]
        sq = np.sqrt(np.maximum(0.0, xo**2 - 1.0))
        res[m_outer] = np.log(xo / 2.0) + (2.0 / sq) * np.arctan(np.sqrt((xo - 1.0) / (1.0 + xo)))

    # x near 1
    m_mid = (~m_inner) & (~m_outer) & (x > 1e-7)
    if np.any(m_mid):
        res[m_mid] = 1.0 + np.log(0.5)

    # x <= 1e-7: smooth Taylor limit g(x) / x -> 0, g(x) -> 0
    m_zero = x <= 1e-7
    if np.any(m_zero):
        res[m_zero] = 0.0

    return res


def _nfw_h(x: np.ndarray) -> np.ndarray:
    """Helper computing the NFW potential radial kernel h(x)."""
    res = np.zeros_like(x, dtype=np.float64)
    # x < 1
    m_inner = (x < 1.0 - 1e-6) & (x > 1e-7)
    if np.any(m_inner):
        xi = x[m_inner]
        res[m_inner] = 0.5 * (np.log(xi / 2.0) ** 2) - 2.0 * (np.arctanh(np.sqrt((1.0 - xi) / (1.0 + xi))) ** 2)

    # x > 1
    m_outer = x > 1.0 + 1e-6
    if np.any(m_outer):
        xo = x[m_outer]
        res[m_outer] = 0.5 * (np.log(xo / 2.0) ** 2) + 2.0 * (np.arctan(np.sqrt((xo - 1.0) / (1.0 + xo))) ** 2)

    # x near 1
    m_mid = (~m_inner) & (~m_outer) & (x > 1e-7)
    if np.any(m_mid):
        res[m_mid] = 0.5 * (np.log(0.5) ** 2)

    # x <= 1e-7
    m_zero = x <= 1e-7
    if np.any(m_zero):
        res[m_zero] = 0.0

    return res


class NFWHalo(BaseLensModel):
    """Navarro-Frenk-White (NFW) spherical dark matter halo lens model.

    Attributes:
        theta_s: Characteristic angular scale radius r_s / D_l (in arcsec or consistent angular units).
        kappa_s: Characteristic dimensionless convergence at scale radius.
        center: (center_x, center_y) coordinates of the dark matter halo centroid.
    """

    def __init__(
        self,
        theta_s: float,
        kappa_s: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialize the NFW dark matter halo model.

        Args:
            theta_s: Scale radius in angular units (> 0).
            kappa_s: Dimensionless characteristic convergence (> 0).
            center: (center_x, center_y) halo centroid.

        Raises:
            ValueError: If theta_s <= 0 or kappa_s <= 0.
        """
        if theta_s <= 0.0:
            raise ValueError(f"NFW scale radius theta_s must be strictly positive, got {theta_s}.")
        if kappa_s <= 0.0:
            raise ValueError(f"NFW characteristic convergence kappa_s must be strictly positive, got {kappa_s}.")

        self._theta_s = float(theta_s)
        self._kappa_s = float(kappa_s)
        self._center = (float(center[0]), float(center[1]))

    @property
    def theta_s(self) -> float:
        """Scale radius in angular units."""
        return self._theta_s

    @property
    def kappa_s(self) -> float:
        """Characteristic convergence at scale radius."""
        return self._kappa_s

    @property
    def center(self) -> Tuple[float, float]:
        """Halo centroid (center_x, center_y)."""
        return self._center

    def deflection(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the 2D reduced deflection angle alpha(theta).

        alpha(x) = 4 * kappa_s * theta_s * (g(x) / x) * (theta / theta_abs)
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        r = np.hypot(dx, dy)

        x = r / self._theta_s
        g_val = _nfw_g(x)

        # alpha_mag = 4 * kappa_s * theta_s * g(x) / x
        # For x > 1e-7: alpha = 4 * kappa_s * theta_s * (g(x) / x) * (dx / r)
        #             = 4 * kappa_s * (g(x) / x^2) * dx
        safe_x2 = np.where(x > 1e-7, x**2, 1.0)
        scale_fac = np.where(x > 1e-7, (4.0 * self._kappa_s) * (g_val / safe_x2), 0.0)

        alpha_x = scale_fac * dx
        alpha_y = scale_fac * dy
        return alpha_x, alpha_y

    def potential(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
        mask_singular: bool = False,
    ) -> np.ndarray:
        """Compute the 2D effective gravitational lensing potential psi(theta).

        psi(x) = 2 * kappa_s * theta_s^2 * h(x)
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        r = np.hypot(dx, dy)

        x = r / self._theta_s
        h_val = _nfw_h(x)
        return (4.0 * self._kappa_s * (self._theta_s**2)) * h_val

    def convergence_analytical(
        self,
        theta_x: Union[float, np.ndarray],
        theta_y: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute the analytical surface mass density convergence kappa(theta).

        kappa(x) = 2 * kappa_s * f(x) / (x^2 - 1)
        """
        tx, ty = validate_coordinates(theta_x, theta_y)
        dx = tx - self._center[0]
        dy = ty - self._center[1]
        r = np.hypot(dx, dy)

        x = r / self._theta_s
        f_val = _nfw_f(x)
        return (2.0 * self._kappa_s) * f_val
